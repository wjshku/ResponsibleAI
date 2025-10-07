#!/usr/bin/env python3
"""
Manipulation Tools Module

This module defines the interface and implementation for the inpainter tool.

Author: AI Assistant
Date: 2024
"""

import os
import cv2
import numpy as np
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod
DEFAULT_PROMPT = "on this car, add minor scratch marks, light surface damage, subtle wear"


class ManipulationTool(ABC):
    """Abstract base class for all manipulation tools"""
    
    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Return the name of the tool"""
        pass
    
    @property
    @abstractmethod
    def tool_version(self) -> str:
        """Return the version of the tool"""
        pass
    
    @abstractmethod
    def process_image(self, image_path: str, mask_path: str, **kwargs) -> str:
        """
        Process an image with the given mask
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
            **kwargs: Additional tool-specific parameters
            
        Returns:
            Path or URL to the processed image
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this tool"""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate tool-specific parameters"""
        return True


class InpainterTool(ManipulationTool):
    """Inpainter tool for car damage using AI APIs"""
    
    def __init__(self, 
                 api_key: str = None, 
                 use_local: bool = False,
                 api_url: str = None,
                 fallback: bool = False,
                 model_name: str = None,
                 model_version: str = None,
                 model_provider: str = None,
                 output_dir: str = None):
        """
        Initialize the inpainter with API credentials and model information
        
        Args:
            api_key: API key for Stability AI (optional if using local)
            use_local: Whether to use local API fallback
            model_name: Name of the AI model to use
            model_version: Version of the AI model
            model_provider: Provider of the AI model
            output_dir: Directory to save processed images (if None, saves next to input)
        """
        self.api_key = api_key
        self.use_local = use_local
        self._tool_name = "inpainter"
        self._tool_version = "1.0.0"
        self.output_dir = output_dir
        self.fallback = fallback
        # Configure API URL
        # If use_local, prefer env INPAINT_API_URL or default local endpoint.
        # If remote (Stability), prefer provided api_url, otherwise use Stability default.
        if use_local:
            self.api_url = api_url or os.environ.get("INPAINT_API_URL", "http://localhost:8000/inpaint")
        else:
            self.api_url = api_url  # may be None; setup_stability_api will set default if needed
        
        # Set model information (use defaults if not provided)
        self._model_name = model_name or (self._get_default_model_name())
        self._model_version = model_version or "1.0.0"
        self._model_provider = model_provider or (self._get_default_model_provider())
        
        if not use_local:
            self.setup_stability_api()
        else:
            print(f"Using local API fallback with model: {self._model_name} v{self._model_version} ({self._model_provider})")
    
    def _get_default_model_name(self) -> str:
        """Get default model name based on mode"""
        if self.use_local:
            return "local-fallback"
        else:
            return "stable-diffusion-xl-1024-v1-0"
    
    def _get_default_model_provider(self) -> str:
        """Get default model provider based on mode"""
        if self.use_local:
            return "local"
        else:
            return "stability-ai"
    
    @property
    def tool_name(self) -> str:
        return self._tool_name
    
    @property
    def tool_version(self) -> str:
        return self._tool_version
    
    @property
    def model_name(self) -> str:
        """Return the name of the AI model used"""
        return self._model_name
    
    @property
    def model_version(self) -> str:
        """Return the version of the AI model used"""
        return self._model_version
    
    @property
    def model_provider(self) -> str:
        """Return the provider of the AI model"""
        return self._model_provider
    
    def setup_stability_api(self):
        """Setup Stability AI API endpoints"""
        if not self.api_key:
            raise ValueError("API key is required for Stability AI")
        # Respect pre-set api_url if provided; otherwise, set default Stability endpoint
        if not getattr(self, 'api_url', None):
            self.api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def inpaint_with_stability(self, image_path: str, mask_path: str, prompt: str = "on this car, add minor scratch marks, light surface damage, subtle wear") -> str:
        """
        Inpaint using Stability AI API
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask image
            prompt: Text prompt for inpainting
            
        Returns:
            URL to the inpainted image
        """
        import requests
        
        # Convert images to base64
        image_b64 = self.image_to_base64(image_path)
        mask_b64 = self.image_to_base64(mask_path)
        
        payload = {
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1
                }
            ],
            "init_image": f"data:image/jpeg;base64,{image_b64}",
            "mask_image": f"data:image/png;base64,{mask_b64}",
            "mask_source": "MASK_IMAGE_WHITE",
            "cfg_scale": 7,
            "steps": 20,
            "samples": 1
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["artifacts"][0]["url"]
    
    def inpaint_with_local_api(self, image_path: str, mask_path: str, prompt: str = "on this car, add minor scratch marks, light surface damage, subtle wear") -> str:
        """
        Inpaint using local API at http://localhost:8000
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask image
            prompt: Text prompt for inpainting
            
        Returns:
            Path to the inpainted image
        """
        import requests
        
        try:
            # Prepare files for upload
            files = {
                'image': ('image.png', open(image_path, 'rb'), 'image/png'),
                'mask': ('mask.png', open(mask_path, 'rb'), 'image/png')
            }
            
            # Prepare data parameters
            data = {
                'prompt': prompt,
                'steps': 20,
                'guidance': 7.5,
                'strength': 0.8,
                'width': 512,
                'height': 512
            }
            
            print(f"📤 Sending request to local API with prompt: {prompt}")
            headers = {}
            if getattr(self, 'api_key', None):
                headers["Authorization"] = f"Bearer {self.api_key}"
            print(f"📤 Sending request to inpaint API: {self.api_url}")
            response = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=60*3)
            response.raise_for_status()
            
            # Close the files
            files['image'][1].close()
            files['mask'][1].close()
            
            # Check if response is an image (PNG format)
            if response.headers.get('content-type') == 'image/png':
                # Save the result image to output directory
                if self.output_dir:
                    # Use output directory if specified
                    os.makedirs(self.output_dir, exist_ok=True)
                    # Generate unique filename with timestamp and random ID
                    import time
                    import random
                    timestamp = int(time.time() * 1000)  # milliseconds
                    random_id = random.randint(1000, 9999)
                    result_path = os.path.join(self.output_dir, f"inpainted_{timestamp}_{random_id}.png")
                else:
                    # Fallback to saving next to input image (for backward compatibility)
                    base_name = os.path.splitext(image_path)[0]
                    if not base_name.endswith('_inpainted'):
                        result_path = f"{base_name}_inpainted.png"
                    else:
                        result_path = f"{base_name}.png"
                
                with open(result_path, "wb") as f:
                    f.write(response.content)
                
                print(f"✅ Local API: Successfully inpainted {os.path.basename(image_path)}")
                print(f"   Result saved to: {result_path}")
                return result_path
            else:
                print(f"❌ Unexpected response format: {response.headers.get('content-type')}")
                return image_path
            
        except Exception as e:
            print(f"❌ Local API error: {e}")
            # Close files if they were opened
            try:
                if 'files' in locals():
                    files['image'][1].close()
                    files['mask'][1].close()
            except:
                pass
            # If we reach here, the API call failed - raise an exception instead of returning None
            raise Exception(f"Local API failed and no fallback available")
    
    
    def process_image(self, image_path: str, mask_path: str, **kwargs) -> str:
        """Process image using inpainting"""
        # Fallback: return original image without processing
        if getattr(self, 'fallback', False):
            return image_path
        prompt = kwargs.get('prompt', DEFAULT_PROMPT)
        
        if self.use_local:
            # Use local API - if it fails, let the exception propagate
            result = self.inpaint_with_local_api(image_path, mask_path, prompt)
        else:
            result = self.inpaint_with_stability(image_path, mask_path, prompt)
        
        # Assert that we got a valid string result
        assert isinstance(result, str), f"process_image must return a string path, got {type(result)}"
        assert result is not None, "process_image returned None"
        assert len(result) > 0, "process_image returned empty string"
        
        return result
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "prompt": DEFAULT_PROMPT,
            "use_local": self.use_local,
            "fallback": getattr(self, 'fallback', False)
        }
    
    def download_result(self, url_or_path: str, output_path: str):
        """Download the inpainted image from URL or copy from local path"""
        if url_or_path.startswith('http'):
            # Download from URL
            import requests
            response = requests.get(url_or_path)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
        else:
            # Copy from local path
            shutil.copy2(url_or_path, output_path)
        
        print(f"Inpainted image saved to: {output_path}")


class EditTool(ManipulationTool):
    """Edit tool that calls an external Image Edit API (e.g., Qwen Image Edit)

    This tool performs prompt-guided edits on the entire image by sending the
    image and parameters to a configured HTTP endpoint that returns the edited
    image (PNG). The provided mask_path is accepted for interface compatibility
    but is not used.
    """
    def __init__(self,
                 use_local: bool = True,
                 api_url: str = None,
                 api_key: str = None,
                 output_dir: str = None,
                 fallback: bool = False,
                 model_name: str = "Qwen Image Edit",
                 model_version: str = "api",
                 model_provider: str = "qwen-api"):
        """
        Initialize EditTool API client
        
        Args:
            api_url: Endpoint for image edit API. Default: http://localhost:8000/edit
            api_key: Optional API key for authentication (sent as Authorization header)
            output_dir: Directory to save edited images (defaults next to input if None)
            model_name: Model display name for metadata
            model_version: Model version label for metadata
            model_provider: Provider label for metadata
        """
        self._tool_name = "editor"
        self._tool_version = "1.1.0"
        self._model_name = model_name
        self._model_version = model_version
        self._model_provider = model_provider
        self.output_dir = output_dir
        self.use_local = use_local
        self.fallback = fallback
        self.api_url = api_url or os.environ.get("EDIT_API_URL", "http://localhost:8000/edit")
        self.api_key = api_key or os.environ.get("EDIT_API_KEY")

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def tool_version(self) -> str:
        return self._tool_version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_provider(self) -> str:
        return self._model_provider

    def process_image(self, image_path: str, mask_path: str = None, **kwargs) -> str:
        """Apply prompt-guided edit via HTTP API.

        Args:
            image_path: Path to the input image.
            mask_path: Unused (for interface compatibility with InpainterTool).
            **kwargs: prompt (str), num_inference_steps (int), true_cfg_scale (float),
                      negative_prompt (str), seed (int)

        Returns:
            Local path to the edited image file.
        """
        prompt = kwargs.get('prompt', DEFAULT_PROMPT)
        num_inference_steps = int(kwargs.get('num_inference_steps', 10))
        true_cfg_scale = float(kwargs.get('true_cfg_scale', 4.0))
        negative_prompt = kwargs.get('negative_prompt', "no change to the undamaged parts of the car in the image")
        seed = kwargs.get('seed', None)

        # Fallback: return original image without processing
        if getattr(self, 'fallback', False):
            return image_path
        # Currently only local/api endpoint flow is supported; delegate here
        return self.edit_with_local_api(
            image_path=image_path,
            mask_path=mask_path,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
        )

    def edit_with_local_api(self,
                             image_path: str,
                             mask_path: str,
                             prompt: str,
                             num_inference_steps: int = 10,
                             true_cfg_scale: float = 2.5,
                             negative_prompt: str = "",
                             seed: int = None) -> str:
        """Send edit request to a local/remote HTTP API and save the PNG result."""
        import requests
        import time
        import random

        # Prepare output path
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            random_id = random.randint(1000, 9999)
            result_path = os.path.join(self.output_dir, f"edited_{timestamp}_{random_id}.png")
        else:
            base_name, _ = os.path.splitext(image_path)
            result_path = f"{base_name}_edited.png"

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        files = {
            'image': ('image.png', open(image_path, 'rb'), 'image/png')
        }

        data = {
            'prompt': prompt,
            'num_inference_steps': num_inference_steps,
            'true_cfg_scale': true_cfg_scale,
            'negative_prompt': negative_prompt,
            'width': 512,
            'height': 512,
        }
        if seed is not None:
            data['seed'] = (int(seed))

        try:
            print(f"📤 Sending edit request to {self.api_url}")
            response = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=60*3)
            response.raise_for_status()
        finally:
            try:
                files['image'][1].close()
            except Exception:
                pass

        if response.headers.get('content-type') == 'image/png':
            with open(result_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Edit API: Successfully edited {os.path.basename(image_path)}")
            print(f"   Result saved to: {result_path}")
            return result_path
        else:
            try:
                print(f"❌ Unexpected response: {response.status_code} {response.text[:200]}")
            except Exception:
                print(f"❌ Unexpected response content-type: {response.headers.get('content-type')}")
            raise RuntimeError("Edit API returned non-image response")

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "prompt": DEFAULT_PROMPT,
            "num_inference_steps": 20,
            "true_cfg_scale": 2.5,
            "negative_prompt": "no change to the undamaged parts of the car in the image, and should not keep the original damage",
            "fallback": getattr(self, 'fallback', False)
        }

    def validate_parameters(self, **kwargs) -> bool:
        try:
            _ = str(kwargs.get('prompt', ''))
            if 'num_inference_steps' in kwargs:
                assert int(kwargs['num_inference_steps']) > 0
            if 'true_cfg_scale' in kwargs:
                float(kwargs['true_cfg_scale'])
            return True
        except Exception:
            return False

def test_inpainter_local_basic():
    """Test case: Basic usage of InpainterTool in local mode with a synthetic mask"""
    print("\n=== Test 1: InpainterTool (local) basic ===")
    try:
        # Create synthetic inputs
        test_image = np.ones((128, 128, 3), dtype=np.uint8) * 200
        cv2.circle(test_image, (64, 64), 30, (255, 0, 0), -1)
        image_path = "inpaint_input.jpg"
        cv2.imwrite(image_path, test_image)

        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(mask, (44, 44), (84, 84), 255, -1)
        mask_path = "inpaint_mask.png"
        cv2.imwrite(mask_path, mask)

        tool = InpainterTool(use_local=True, output_dir="./api_debug", fallback=True)
        out_path = tool.process_image(
                        image_path, 
                        mask_path, 
                        prompt='on this car, add minor scratch marks, light surface damage, subtle wear'
                        )
        print(f"✓ Inpainter local produced: {out_path}")
    except Exception as e:
        print(f"✗ Inpainter local test failed: {e}")


def test_edit_tool_basic():
    """Test case: Basic usage of EditTool (Qwen Image Edit) on a small image"""
    print("\n=== Test 2: EditTool (Qwen Image Edit) basic ===")
    try:
        # Create synthetic inputs
        test_image = np.ones((128, 128, 3), dtype=np.uint8) * 200
        cv2.circle(test_image, (64, 64), 30, (255, 0, 0), -1)
        image_path = "./edit_input.jpg"
        cv2.imwrite(image_path, test_image)

        tool = EditTool(use_local=True, output_dir="./api_debug", fallback=False)
        out_path = tool.process_image(image_path, 
            prompt='on this car, add minor scratch marks, light surface damage, subtle wear', 
            num_inference_steps=10,
            negative_prompt="no change to the undamaged parts of the car in the image")
        print(f"✓ EditTool local produced: {out_path}")
    except Exception as e:
        print(f"✗ Edit basic test failed: {e}")

def main():
    test_edit_tool_basic()


if __name__ == "__main__":
    main()