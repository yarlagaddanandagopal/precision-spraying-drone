"""
Manual Retraining Trigger
Run this script to manually trigger the automated retraining pipeline
"""
from pipeline_orchestrator import PipelineOrchestrator

def main():
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  🚀 MANUAL RETRAINING TRIGGER".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝\n")
    
    orchestrator = PipelineOrchestrator()
    
    # Check current status
    print("📊 Checking current status...\n")
    orchestrator.print_status()
    
    # Ask user to confirm
    response = input("\nDo you want to start retraining? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        print("\n🚀 Starting automated retraining pipeline...\n")
        success = orchestrator.run_full_pipeline()
        
        if success:
            print("\n✅ RETRAINING COMPLETED!")
            print("🔄 Restart main_v2.py to use the new model")
        else:
            print("\n❌ RETRAINING FAILED")
            print("Check the logs for details")
    else:
        print("\n❌ Cancelled")

if __name__ == "__main__":
    main()
