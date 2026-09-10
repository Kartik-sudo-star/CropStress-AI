"""
End-to-End Tests

These tests require the full system to be running (backend + frontend).
Run with: pytest tests/e2e/ -v
"""
import pytest
from playwright.sync_api import sync_playwright
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# These tests require the backend and frontend servers to be running
# They are marked as skipped by default


@pytest.mark.skip(reason="Requires running servers")
class TestE2EWorkflow:
    """End-to-end workflow tests."""
    
    @classmethod
    def setup_class(cls):
        """Start backend and frontend servers."""
        # In practice, you'd start these separately
        # cls.backend_process = subprocess.Popen(["python", "-m", "backend.app.main"])
        # cls.frontend_process = subprocess.Popen(["npm", "run", "dev"], cwd="frontend")
        # time.sleep(5)  # Wait for servers to start
        pass
    
    @classmethod
    def teardown_class(cls):
        """Stop servers."""
        # cls.backend_process.terminate()
        # cls.frontend_process.terminate()
        pass
    
    def test_home_page_loads(self):
        """Test home page loads correctly."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:5173")
            
            # Check for main title
            assert "Deepfake Detection" in page.content()
            
            browser.close()
    
    def test_upload_and_analyze_image(self):
        """Test uploading an image and getting results."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:5173")
            
            # Navigate to analyze page
            page.click("text=START ANALYSIS")
            page.wait_for_url("**/analyze")
            
            # Upload file
            page.set_input_files('input[type="file"]', "tests/fixtures/test_image.jpg")
            
            # Click analyze
            page.click("text=ANALYZE MEDIA")
            
            # Wait for processing
            page.wait_for_selector("text=ANALYSIS COMPLETE", timeout=60000)
            
            # Check results
            assert "POTENTIALLY MANIPULATED" in page.content() or "LIKELY AUTHENTIC" in page.content()
            
            browser.close()


@pytest.mark.skip(reason="Requires test fixtures")
class TestE2EFixtures:
    """Tests with specific test fixtures."""
    
    def test_known_deepfake_detection(self):
        """Test detection on known deepfake sample."""
        pass
    
    def test_known_authentic_detection(self):
        """Test detection on known authentic sample."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])