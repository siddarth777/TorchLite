import unittest
import sys

if __name__ == "__main__":
    loader = unittest.TestLoader()
    # Discover all tests in the current directory matching 'test_*.py'
    suite = loader.discover(start_dir='.', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)
