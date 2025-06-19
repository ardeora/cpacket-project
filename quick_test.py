#!/usr/bin/env python3
"""
Quick test script for the DDos Classifier - trains a single model using one feature selection method.
This is useful for testing the setup before running the full training pipeline.
"""

from classifier import DDosClassifier
import sys

def quick_test():
    """Run a quick test with one feature selection method."""
    print("🧪 Quick Test: Training Random Forest with Information Gain features")
    print("="*60)
    
    try:
        # Initialize classifier
        classifier = DDosClassifier()
        
        # Load data
        classifier.load_data()
        
        # Load feature selection methods
        classifier.load_feature_methods()
        
        # Train a single model using information_gain features
        method_name = 'information_gain'
        
        if method_name not in classifier.feature_methods:
            print(f"❌ Method '{method_name}' not found.")
            print(f"Available methods: {list(classifier.feature_methods.keys())}")
            return
        
        print(f"\n🎯 Training model using '{method_name}' features...")
        
        # Get features for this method (limit to 20 for quick test)
        features = classifier.get_features_for_method(method_name, max_features=20)
        
        if not features:
            print(f"❌ No valid features found for method '{method_name}'")
            return
        
        # Prepare data
        X_train, X_test, y_train, y_test = classifier.prepare_data(features)
        
        # Train model with fewer estimators for quick test
        model = classifier.train_random_forest(
            X_train, y_train,
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            random_state=42
        )
        
        # Evaluate model
        results = classifier.evaluate_model(model, X_test, y_test, method_name)
        
        # Save model
        classifier.save_model(model, f"{method_name}_quick_test", features, results)
        
        print(f"\n✅ Quick test completed successfully!")
        print(f"📊 Accuracy: {results['accuracy']:.4f}")
        print(f"💾 Model saved as: rf_classifier_{method_name}_quick_test.joblib")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during quick test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check if user wants to run specific method
    if len(sys.argv) > 1:
        method = sys.argv[1]
        print(f"Testing with method: {method}")
        # Could add logic here to test specific method
    
    quick_test()
