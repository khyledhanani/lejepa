"""
Helper script to run comparison experiments between continuous and discrete modes
"""
import subprocess
import sys
from pathlib import Path

def run_experiment(config_name, description):
    """Run a single experiment with given config"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print(f"Config: {config_name}")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable,
        "funstuf.py/fun.py",
        f"--config-name={config_name}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {description}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted: {description}")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║        Discrete Bottleneck Multi-View Learning                 ║
║              Experiment Runner                                 ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    experiments = [
        ("config_continuous", "Baseline: Continuous SIGReg"),
        ("config_discrete", "Discrete: 8 groups × 16 categories"),
        ("config_discrete_large", "Discrete: 16 groups × 16 categories"),
    ]
    
    print("Available experiments:")
    for i, (config, desc) in enumerate(experiments, 1):
        print(f"  {i}. {desc}")
    print(f"  {len(experiments)+1}. Run all sequentially")
    print("  0. Exit")
    
    choice = input("\nSelect experiment to run (0-{}): ".format(len(experiments)+1))
    
    try:
        choice = int(choice)
    except ValueError:
        print("Invalid choice!")
        return
    
    if choice == 0:
        print("Exiting...")
        return
    elif choice == len(experiments) + 1:
        # Run all
        results = []
        for config, desc in experiments:
            success = run_experiment(config, desc)
            results.append((desc, success))
        
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        for desc, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status}: {desc}")
    elif 1 <= choice <= len(experiments):
        config, desc = experiments[choice - 1]
        run_experiment(config, desc)
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*70)
    print("View results in W&B: https://wandb.ai/")
    print("Saved models in: ./saved_models/")
    print("="*70)

if __name__ == "__main__":
    main()

