#![feature(allocator_api)]

use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::time::Instant;

use bellman::kate_commitment::{Crs, CrsForMonomialForm};
use bellman::worker::Worker as BellmanWorker;

use zkos_wrapper::{
    Bn256, L1_VERIFIER_DOMAIN_SIZE_LOG,
    calculate_verification_key_hash, deserialize_from_file, get_trusted_setup, serialize_to_file,
    circuits::{BinaryCommitment, RiscWrapperWitness},
};

#[derive(Parser)]
#[command(
    name = "wrapper",
    version,
    about = "Wrap FRI proofs from zksync-airbender into SNARKs verifiable on Ethereum"
)]
struct Cli {
    /// Number of worker threads (defaults to all available cores)
    #[arg(long, global = true)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Full pipeline: FRI proof -> SNARK proof (phases 1+2+3)
    ProveAll {
        /// Path to the FRI proof JSON (UnrolledProgramProof from airbender)
        #[arg(long)]
        proof: PathBuf,

        /// Path to the RISC-V .bin file (omit to use default recursion verifier)
        #[arg(long, requires = "text")]
        bin: Option<PathBuf>,

        /// Path to the RISC-V .text file (omit to use default recursion verifier)
        #[arg(long, requires = "bin")]
        text: Option<PathBuf>,

        /// Output directory for proof and VK files
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Path to trusted setup (CRS) file. If omitted, uses fake crs_42 for testing.
        #[arg(long)]
        trusted_setup: Option<PathBuf>,

        /// Enable zero-knowledge padding in SNARK proving phase
        #[arg(long)]
        use_zk: bool,

        /// Save intermediate proofs (risc wrapper, compression) alongside final SNARK
        #[arg(long)]
        save_intermediates: bool,
    },

    /// Phase 1: Wrap FRI proof into a boojum STARK proof
    ProveRiscWrapper {
        /// Path to the FRI proof JSON (UnrolledProgramProof)
        #[arg(long)]
        proof: PathBuf,

        /// Path to the RISC-V .bin file (omit to use default recursion verifier)
        #[arg(long, requires = "text")]
        bin: Option<PathBuf>,

        /// Path to the RISC-V .text file (omit to use default recursion verifier)
        #[arg(long, requires = "bin")]
        text: Option<PathBuf>,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,
    },

    /// Phase 2: Compress a STARK wrapper proof (Poseidon2-based re-hashing)
    ProveCompression {
        /// Path to risc_wrapper_proof.json
        #[arg(long)]
        risc_wrapper_proof: PathBuf,

        /// Path to risc_wrapper_vk.json
        #[arg(long)]
        risc_wrapper_vk: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,
    },

    /// Phase 3: Wrap compressed STARK proof into a BN256 SNARK
    ProveSnark {
        /// Path to compression_proof.json
        #[arg(long)]
        compression_proof: PathBuf,

        /// Path to compression_vk.json
        #[arg(long)]
        compression_vk: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Path to trusted setup (CRS) file. If omitted, uses fake crs_42 for testing.
        #[arg(long)]
        trusted_setup: Option<PathBuf>,

        /// Enable zero-knowledge padding in SNARK proving
        #[arg(long)]
        use_zk: bool,
    },

    /// Generate verification keys without a proof (for deployment preparation)
    GenerateVk {
        /// Output directory for VK files
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Path to RISC-V .bin file (omit to use default recursion verifier)
        #[arg(long, requires = "text")]
        bin: Option<PathBuf>,

        /// Path to RISC-V .text file (omit to use default recursion verifier)
        #[arg(long, requires = "bin")]
        text: Option<PathBuf>,

        /// Path to trusted setup (CRS) file. If omitted, uses fake crs_42.
        #[arg(long)]
        trusted_setup: Option<PathBuf>,
    },

    /// Compute the Keccak256 hash of a SNARK verification key
    VkHash {
        /// Path to snark_vk.json
        #[arg(long)]
        vk: PathBuf,
    },

    /// Verify a proof at any pipeline stage
    Verify {
        /// Pipeline stage of the proof
        #[arg(long, value_enum)]
        stage: VerifyStage,

        /// Path to the proof JSON file
        #[arg(long)]
        proof: PathBuf,

        /// Path to the verification key JSON file
        #[arg(long)]
        vk: PathBuf,
    },
}

#[derive(Clone, ValueEnum)]
enum VerifyStage {
    RiscWrapper,
    Compression,
    Snark,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_boojum_worker(threads: Option<usize>) -> boojum::worker::Worker {
    match threads {
        Some(n) => {
            println!("Using {n} worker threads");
            boojum::worker::Worker::new_with_num_threads(n)
        }
        None => boojum::worker::Worker::new(),
    }
}

fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    println!("=== {label}: starting...");
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    println!("=== {label}: completed in {:.1}s", elapsed.as_secs_f64());
    result
}

fn load_binary_commitment(
    bin: &Option<PathBuf>,
    text: &Option<PathBuf>,
) -> Result<BinaryCommitment, Box<dyn std::error::Error>> {
    match (bin, text) {
        (Some(bin_path), Some(text_path)) => {
            println!("Loading binary from {}", bin_path.display());
            let binary = std::fs::read(bin_path)
                .map_err(|e| format!("Failed to read .bin file {}: {e}", bin_path.display()))?;
            let text_data = std::fs::read(text_path)
                .map_err(|e| format!("Failed to read .text file {}: {e}", text_path.display()))?;
            Ok(BinaryCommitment::from_binary(&binary, &text_data))
        }
        _ => {
            println!("Using default recursion verifier binary");
            Ok(BinaryCommitment::from_default_binary())
        }
    }
}

fn ensure_output_dir(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(path)
        .map_err(|e| format!("Failed to create output directory {}: {e}", path.display()))?;
    Ok(())
}

fn load_crs(trusted_setup: &Option<PathBuf>) -> Crs<Bn256, CrsForMonomialForm> {
    match trusted_setup {
        Some(path) => {
            println!("Loading trusted setup from {}", path.display());
            get_trusted_setup(&path.to_string_lossy().to_string())
        }
        None => {
            println!("WARNING: Using fake crs_42 trusted setup (testing only, NOT for production!)");
            Crs::<Bn256, CrsForMonomialForm>::crs_42(
                1 << L1_VERIFIER_DOMAIN_SIZE_LOG,
                &BellmanWorker::new(),
            )
        }
    }
}

fn output_path(dir: &Path, filename: &str) -> String {
    dir.join(filename).to_string_lossy().to_string()
}

// ---------------------------------------------------------------------------
// Phase implementations
// ---------------------------------------------------------------------------

fn run_phase1_risc_wrapper(
    proof_path: &Path,
    bin: &Option<PathBuf>,
    text: &Option<PathBuf>,
    worker: &boojum::worker::Worker,
) -> Result<
    (zkos_wrapper::RiscWrapperProof, zkos_wrapper::RiscWrapperVK),
    Box<dyn std::error::Error>,
> {
    let binary_commitment = timed("Phase 1 - binary commitment", || {
        load_binary_commitment(bin, text)
    })?;

    println!("Loading FRI proof from {}", proof_path.display());
    let program_proof: execution_utils::unrolled::UnrolledProgramProof =
        deserialize_from_file(proof_path.to_str().unwrap());

    let risc_wrapper_witness = timed("Phase 1 - witness generation", || {
        RiscWrapperWitness::from_full_proof(program_proof, &binary_commitment)
    });

    let (finalization_hint, setup_base, setup, risc_wrapper_vk, setup_tree, vars_hint, witness_hints) =
        timed("Phase 1 - setup", || {
            zkos_wrapper::get_risc_wrapper_setup(worker, binary_commitment)
        });

    let risc_wrapper_proof = timed("Phase 1 - prove", || {
        zkos_wrapper::prove_risc_wrapper(
            risc_wrapper_witness,
            &finalization_hint,
            &setup_base,
            &setup,
            &risc_wrapper_vk,
            &setup_tree,
            &vars_hint,
            &witness_hints,
            worker,
            binary_commitment,
        )
    });

    let is_valid = timed("Phase 1 - verify", || {
        zkos_wrapper::verify_risc_wrapper_proof(&risc_wrapper_proof, &risc_wrapper_vk)
    });
    if !is_valid {
        return Err("RISC wrapper proof verification failed".into());
    }
    println!("Phase 1 proof verified successfully");

    Ok((risc_wrapper_proof, risc_wrapper_vk))
}

fn run_phase2_compression(
    risc_wrapper_proof: zkos_wrapper::RiscWrapperProof,
    risc_wrapper_vk: zkos_wrapper::RiscWrapperVK,
    worker: &boojum::worker::Worker,
) -> Result<
    (zkos_wrapper::CompressionProof, zkos_wrapper::CompressionVK),
    Box<dyn std::error::Error>,
> {
    let (finalization_hint, setup_base, setup, compression_vk, setup_tree, vars_hint, witness_hints) =
        timed("Phase 2 - setup", || {
            zkos_wrapper::get_compression_setup(risc_wrapper_vk.clone(), worker)
        });

    let compression_proof = timed("Phase 2 - prove", || {
        zkos_wrapper::prove_compression(
            risc_wrapper_proof,
            risc_wrapper_vk,
            &finalization_hint,
            &setup_base,
            &setup,
            &compression_vk,
            &setup_tree,
            &vars_hint,
            &witness_hints,
            worker,
        )
    });

    let is_valid = timed("Phase 2 - verify", || {
        zkos_wrapper::verify_compression_proof(&compression_proof, &compression_vk)
    });
    if !is_valid {
        return Err("Compression proof verification failed".into());
    }
    println!("Phase 2 proof verified successfully");

    Ok((compression_proof, compression_vk))
}

fn run_phase3_snark(
    compression_proof: zkos_wrapper::CompressionProof,
    compression_vk: zkos_wrapper::CompressionVK,
    trusted_setup: &Option<PathBuf>,
    use_zk: bool,
) -> Result<
    (zkos_wrapper::SnarkWrapperProof, zkos_wrapper::SnarkWrapperVK),
    Box<dyn std::error::Error>,
> {
    let crs_mons = timed("Phase 3 - load CRS", || load_crs(trusted_setup));

    let bellman_worker = BellmanWorker::new();

    let (snark_setup, snark_vk) = timed("Phase 3 - setup", || {
        zkos_wrapper::get_snark_wrapper_setup(compression_vk.clone(), &crs_mons, &bellman_worker)
    });

    let snark_proof = timed("Phase 3 - prove", || {
        zkos_wrapper::prove_snark_wrapper(
            compression_proof,
            compression_vk,
            &snark_setup,
            &crs_mons,
            &bellman_worker,
            use_zk,
        )
    });

    let is_valid = timed("Phase 3 - verify", || {
        zkos_wrapper::verify_snark_wrapper_proof(&snark_proof, &snark_vk)
    });
    if !is_valid {
        return Err("SNARK wrapper proof verification failed".into());
    }
    println!("Phase 3 proof verified successfully");

    Ok((snark_proof, snark_vk))
}

// ---------------------------------------------------------------------------
// Subcommand handlers
// ---------------------------------------------------------------------------

fn cmd_prove_all(
    proof: PathBuf,
    bin: Option<PathBuf>,
    text: Option<PathBuf>,
    output_dir: PathBuf,
    trusted_setup: Option<PathBuf>,
    use_zk: bool,
    save_intermediates: bool,
    threads: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_output_dir(&output_dir)?;
    let total_start = Instant::now();
    let worker = create_boojum_worker(threads);

    // Phase 1
    let (risc_wrapper_proof, risc_wrapper_vk) =
        run_phase1_risc_wrapper(&proof, &bin, &text, &worker)?;

    if save_intermediates {
        serialize_to_file(&risc_wrapper_proof, &output_path(&output_dir, "risc_wrapper_proof.json"));
        serialize_to_file(&risc_wrapper_vk, &output_path(&output_dir, "risc_wrapper_vk.json"));
        println!("Saved intermediate Phase 1 outputs");
    }

    // Phase 2
    let (compression_proof, compression_vk) =
        run_phase2_compression(risc_wrapper_proof, risc_wrapper_vk, &worker)?;

    if save_intermediates {
        serialize_to_file(&compression_proof, &output_path(&output_dir, "compression_proof.json"));
        serialize_to_file(&compression_vk, &output_path(&output_dir, "compression_vk.json"));
        println!("Saved intermediate Phase 2 outputs");
    }

    // Phase 3
    let (snark_proof, snark_vk) =
        run_phase3_snark(compression_proof, compression_vk, &trusted_setup, use_zk)?;

    serialize_to_file(&snark_proof, &output_path(&output_dir, "snark_proof.json"));
    serialize_to_file(&snark_vk, &output_path(&output_dir, "snark_vk.json"));

    let vk_hash = calculate_verification_key_hash(snark_vk);
    println!("SNARK VK hash: {vk_hash:?}");

    let total_elapsed = total_start.elapsed();
    println!("=== Total pipeline time: {:.1}s", total_elapsed.as_secs_f64());

    Ok(())
}

fn cmd_prove_risc_wrapper(
    proof: PathBuf,
    bin: Option<PathBuf>,
    text: Option<PathBuf>,
    output_dir: PathBuf,
    threads: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_output_dir(&output_dir)?;
    let worker = create_boojum_worker(threads);

    let (risc_wrapper_proof, risc_wrapper_vk) =
        run_phase1_risc_wrapper(&proof, &bin, &text, &worker)?;

    serialize_to_file(&risc_wrapper_proof, &output_path(&output_dir, "risc_wrapper_proof.json"));
    serialize_to_file(&risc_wrapper_vk, &output_path(&output_dir, "risc_wrapper_vk.json"));

    Ok(())
}

fn cmd_prove_compression(
    risc_wrapper_proof_path: PathBuf,
    risc_wrapper_vk_path: PathBuf,
    output_dir: PathBuf,
    threads: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_output_dir(&output_dir)?;
    let worker = create_boojum_worker(threads);

    println!("Loading RISC wrapper proof from {}", risc_wrapper_proof_path.display());
    let risc_wrapper_proof = deserialize_from_file(risc_wrapper_proof_path.to_str().unwrap());
    println!("Loading RISC wrapper VK from {}", risc_wrapper_vk_path.display());
    let risc_wrapper_vk = deserialize_from_file(risc_wrapper_vk_path.to_str().unwrap());

    let (compression_proof, compression_vk) =
        run_phase2_compression(risc_wrapper_proof, risc_wrapper_vk, &worker)?;

    serialize_to_file(&compression_proof, &output_path(&output_dir, "compression_proof.json"));
    serialize_to_file(&compression_vk, &output_path(&output_dir, "compression_vk.json"));

    Ok(())
}

fn cmd_prove_snark(
    compression_proof_path: PathBuf,
    compression_vk_path: PathBuf,
    output_dir: PathBuf,
    trusted_setup: Option<PathBuf>,
    use_zk: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_output_dir(&output_dir)?;

    println!("Loading compression proof from {}", compression_proof_path.display());
    let compression_proof = deserialize_from_file(compression_proof_path.to_str().unwrap());
    println!("Loading compression VK from {}", compression_vk_path.display());
    let compression_vk = deserialize_from_file(compression_vk_path.to_str().unwrap());

    let (snark_proof, snark_vk) =
        run_phase3_snark(compression_proof, compression_vk, &trusted_setup, use_zk)?;

    serialize_to_file(&snark_proof, &output_path(&output_dir, "snark_proof.json"));
    serialize_to_file(&snark_vk, &output_path(&output_dir, "snark_vk.json"));

    let vk_hash = calculate_verification_key_hash(snark_vk);
    println!("SNARK VK hash: {vk_hash:?}");

    Ok(())
}

fn cmd_generate_vk(
    output_dir: PathBuf,
    bin: Option<PathBuf>,
    text: Option<PathBuf>,
    trusted_setup: Option<PathBuf>,
    threads: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_output_dir(&output_dir)?;
    let worker = create_boojum_worker(threads);

    // Phase 1: RISC wrapper VK
    let binary_commitment = timed("VK generation - binary commitment", || {
        load_binary_commitment(&bin, &text)
    })?;

    let (_, _, _, risc_wrapper_vk, _, _, _) = timed("VK generation - Phase 1 (RISC wrapper)", || {
        zkos_wrapper::get_risc_wrapper_setup(&worker, binary_commitment)
    });
    serialize_to_file(&risc_wrapper_vk, &output_path(&output_dir, "risc_wrapper_vk.json"));
    println!("Saved risc_wrapper_vk.json");

    // Phase 2: Compression VK
    let (_, _, _, compression_vk, _, _, _) = timed("VK generation - Phase 2 (compression)", || {
        zkos_wrapper::get_compression_setup(risc_wrapper_vk, &worker)
    });
    serialize_to_file(&compression_vk, &output_path(&output_dir, "compression_vk.json"));
    println!("Saved compression_vk.json");

    // Phase 3: SNARK VK
    let crs_mons = timed("VK generation - load CRS", || load_crs(&trusted_setup));
    let bellman_worker = BellmanWorker::new();

    let (_, snark_vk) = timed("VK generation - Phase 3 (SNARK)", || {
        zkos_wrapper::get_snark_wrapper_setup(compression_vk, &crs_mons, &bellman_worker)
    });
    serialize_to_file(&snark_vk, &output_path(&output_dir, "snark_vk.json"));
    println!("Saved snark_vk.json");

    let vk_hash = calculate_verification_key_hash(snark_vk);
    println!("SNARK VK hash: {vk_hash:?}");

    Ok(())
}

fn cmd_vk_hash(vk_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading VK from {}", vk_path.display());
    let vk = deserialize_from_file(vk_path.to_str().unwrap());
    let vk_hash = calculate_verification_key_hash(vk);
    println!("SNARK VK hash: {vk_hash:?}");
    Ok(())
}

fn cmd_verify(
    stage: VerifyStage,
    proof_path: PathBuf,
    vk_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let proof_str = proof_path.to_str().unwrap();
    let vk_str = vk_path.to_str().unwrap();

    let is_valid = match stage {
        VerifyStage::RiscWrapper => {
            println!("Verifying RISC wrapper proof...");
            let proof = deserialize_from_file(proof_str);
            let vk = deserialize_from_file(vk_str);
            zkos_wrapper::verify_risc_wrapper_proof(&proof, &vk)
        }
        VerifyStage::Compression => {
            println!("Verifying compression proof...");
            let proof = deserialize_from_file(proof_str);
            let vk = deserialize_from_file(vk_str);
            zkos_wrapper::verify_compression_proof(&proof, &vk)
        }
        VerifyStage::Snark => {
            println!("Verifying SNARK proof...");
            let proof = deserialize_from_file(proof_str);
            let vk = deserialize_from_file(vk_str);
            zkos_wrapper::verify_snark_wrapper_proof(&proof, &vk)
        }
    };

    if is_valid {
        println!("Proof is VALID");
        Ok(())
    } else {
        Err("Proof verification FAILED".into())
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ProveAll {
            proof,
            bin,
            text,
            output_dir,
            trusted_setup,
            use_zk,
            save_intermediates,
        } => cmd_prove_all(proof, bin, text, output_dir, trusted_setup, use_zk, save_intermediates, cli.threads),

        Commands::ProveRiscWrapper {
            proof,
            bin,
            text,
            output_dir,
        } => cmd_prove_risc_wrapper(proof, bin, text, output_dir, cli.threads),

        Commands::ProveCompression {
            risc_wrapper_proof,
            risc_wrapper_vk,
            output_dir,
        } => cmd_prove_compression(risc_wrapper_proof, risc_wrapper_vk, output_dir, cli.threads),

        Commands::ProveSnark {
            compression_proof,
            compression_vk,
            output_dir,
            trusted_setup,
            use_zk,
        } => cmd_prove_snark(compression_proof, compression_vk, output_dir, trusted_setup, use_zk),

        Commands::GenerateVk {
            output_dir,
            bin,
            text,
            trusted_setup,
        } => cmd_generate_vk(output_dir, bin, text, trusted_setup, cli.threads),

        Commands::VkHash { vk } => cmd_vk_hash(vk),

        Commands::Verify { stage, proof, vk } => cmd_verify(stage, proof, vk),
    }
}
