import asyncio
import aiohttp
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from datetime import datetime

class VHDLDescriptionGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b", max_generations_per_json: int = 10):
        self.ollama_url = ollama_url
        self.model = model
        self.max_generations_per_json = max_generations_per_json
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.batch_counter = 0
    
    def load_vhdl_from_json(self, json_file_path: str) -> List[Dict[str, str]]:
        """Load VHDL snippets from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            vhdl_files = []
            if isinstance(data, list):
                for item in data:
                    vhdl_files.append({
                        "id": item.get("file_name", f"snippet_{len(vhdl_files)}"),
                        "code": item.get("content", "")
                    })
            else:
                vhdl_files.append({
                    "id": data.get("file_name", "snippet_0"),
                    "code": data.get("content", "")
                })
            
            print(f"✅ Loaded {len(vhdl_files)} VHDL snippets from {json_file_path}")
            return vhdl_files
            
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return []
    
    def create_prompts(self, vhd_code: str) -> Dict[str, str]:
        """Create the three different prompt types for VHDL code analysis"""
        return {
            "high_level_global_summary": f"Provide a high-level, brief summary of the following VHDL code.\nThe summary should describe the main purpose and functionality of the module.\n\nVHDL code:\n---\n{vhd_code}\n---",
            "detailed_global_summary": f"Provide a detailed summary of the following VHDL code.\nThe summary should describe the functionality, inputs, outputs, and internal signals and processes of the VHDL module in detail.\n\nVHDL code:\n---\n{vhd_code}\n---",
            "block_summary": f"Provide a detailed description of the following VHDL code, focusing on describing the code by blocks.\nBreak down the code into logical blocks (e.g., state machine states, processes, combinational logic) and describe each block's function.\n\nVHDL code:\n---\n{vhd_code}\n---"
        }
    
    async def generate_single_description(self, session: aiohttp.ClientSession, prompt: str, description_type: str) -> Dict[str, Any]:
        """Generate a single description using Ollama API"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        
        try:
            async with session.post(f"{self.ollama_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "type": description_type,
                        "description": result.get("response", ""),
                        "success": True,
                        "error": None
                    }
                else:
                    return {
                        "type": description_type,
                        "description": "",
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "type": description_type,
                "description": "",
                "success": False,
                "error": str(e)
            }
    
    async def process_single_vhdl_file(self, session: aiohttp.ClientSession, vhd_code: str, file_id: str) -> Dict[str, Any]:
        """Process a single VHDL file and generate all three descriptions concurrently"""
        prompts = self.create_prompts(vhd_code)
        
        tasks = [
            self.generate_single_description(session, prompt, desc_type)
            for desc_type, prompt in prompts.items()
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        descriptions = {}
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif result["success"]:
                descriptions[result["type"]] = result["description"]
            else:
                errors.append(f"{result['type']}: {result['error']}")
        
        return {
            "file_id": file_id,
            "vhdl_code": vhd_code,
            "descriptions": descriptions,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "errors": errors,
            "success": len(descriptions) == 3
        }
    
    def save_batch_results(self, batch_results: List[Dict[str, Any]], batch_number: int, total_batches: int) -> str:
        """Save a batch of results to a separate JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"vhdl_descriptions_batch_{batch_number:03d}_of_{total_batches:03d}_{timestamp}.json"
        output_path = self.output_dir / output_filename
        
        # Calculate batch statistics
        successful_files = sum(1 for r in batch_results if r["success"])
        total_processing_time = sum(r.get("processing_time", 0) for r in batch_results)
        
        batch_data = {
            "batch_metadata": {
                "batch_number": batch_number,
                "total_batches": total_batches,
                "files_in_batch": len(batch_results),
                "successful_files": successful_files,
                "failed_files": len(batch_results) - successful_files,
                "batch_processing_time": total_processing_time,
                "average_time_per_file": total_processing_time / len(batch_results) if batch_results else 0,
                "generation_timestamp": datetime.now().isoformat(),
                "model_used": self.model
            },
            "results": batch_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Batch {batch_number}/{total_batches} saved: {output_filename}")
        print(f"   ✅ Success: {successful_files}/{len(batch_results)} files")
        print(f"   ⏱️  Time: {total_processing_time:.2f}s")
        
        # Show any errors in this batch
        failed_files = [r for r in batch_results if not r["success"]]
        if failed_files:
            print(f"   ❌ Failed files in batch {batch_number}:")
            for failed in failed_files[:3]:  # Show first 3 failures
                print(f"      - {failed['file_id']}: {failed.get('error', 'Unknown error')}")
            if len(failed_files) > 3:
                print(f"      ... and {len(failed_files) - 3} more")
        
        return str(output_path)
    
    async def process_batch(self, vhdl_batch: List[Dict[str, str]], batch_number: int, total_batches: int, max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Process a single batch of VHDL files"""
        print(f"\n🚀 Processing batch {batch_number}/{total_batches} ({len(vhdl_batch)} files)")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(session, vhd_code, file_id):
            async with semaphore:
                return await self.process_single_vhdl_file(session, vhd_code, file_id)
        
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=300)
        
        batch_start_time = time.time()
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                process_with_semaphore(session, file_data["code"], file_data["id"])
                for file_data in vhdl_batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "file_id": vhdl_batch[i]["id"],
                        "success": False,
                        "error": str(result),
                        "descriptions": {},
                        "processing_time": 0,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    processed_results.append(result)
        
        batch_time = time.time() - batch_start_time
        print(f"⏱️  Batch {batch_number} completed in {batch_time:.2f}s")
        
        # Save this batch immediately
        self.save_batch_results(processed_results, batch_number, total_batches)
        
        return processed_results
    
    async def process_all_vhdl_files_in_batches(self, vhdl_files: List[Dict[str, str]], max_concurrent: int = 3) -> Dict[str, Any]:
        """Process all VHDL files in batches, saving each batch separately"""
        total_files = len(vhdl_files)
        total_batches = (total_files + self.max_generations_per_json - 1) // self.max_generations_per_json
        
        print(f"📊 Processing {total_files} files in {total_batches} batches")
        print(f"📁 Max files per batch: {self.max_generations_per_json}")
        print(f"🔄 Max concurrent requests: {max_concurrent}")
        print(f"💾 Output directory: {self.output_dir}")
        
        all_batch_files = []
        overall_start_time = time.time()
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.max_generations_per_json
            end_idx = min(start_idx + self.max_generations_per_json, total_files)
            batch = vhdl_files[start_idx:end_idx]
            
            batch_results = await self.process_batch(batch, batch_num + 1, total_batches, max_concurrent)
            all_batch_files.append({
                "batch_number": batch_num + 1,
                "batch_file": f"vhdl_descriptions_batch_{batch_num + 1:03d}_of_{total_batches:03d}",
                "files_processed": len(batch_results),
                "successful_files": sum(1 for r in batch_results if r["success"])
            })
        
        total_time = time.time() - overall_start_time
        
        # Create summary file
        summary = {
            "processing_summary": {
                "total_files": total_files,
                "total_batches": total_batches,
                "max_files_per_batch": self.max_generations_per_json,
                "total_processing_time": total_time,
                "average_time_per_file": total_time / total_files if total_files > 0 else 0,
                "completion_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "max_concurrent": max_concurrent
            },
            "batch_files": all_batch_files
        }
        
        summary_path = self.output_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 ALL PROCESSING COMPLETED!")
        print(f"📊 Total time: {total_time:.2f}s")
        print(f"📁 Summary saved: {summary_path}")
        
        return summary

# Check if Ollama is running and test connection
async def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    models = await response.json()
                    print("✅ Ollama is running!")
                    available_models = [m['name'] for m in models.get('models', [])]
                    print(f"📋 Available models: {available_models}")
                    return True
                else:
                    print(f"❌ Ollama responded with status {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

# Main function to process your JSON file in batches
async def process_hdl_coder_json_in_batches(
    json_file_path: str = "hdl_coder_20.json", 
    max_generations_per_json: int = 10, 
    max_concurrent: int = 3
):
    """
    Process VHDL snippets from JSON file in batches
    
    Args:
        json_file_path: Path to your JSON file
        max_generations_per_json: Maximum number of generations per output JSON file
        max_concurrent: Maximum number of concurrent requests
    """
    print(f"🚀 Starting batch processing...")
    print(f"📁 Input file: {json_file_path}")
    print(f"📊 Max generations per JSON: {max_generations_per_json}")
    
    generator = VHDLDescriptionGenerator(max_generations_per_json=max_generations_per_json)
    
    # Load VHDL snippets from JSON
    vhdl_files = generator.load_vhdl_from_json(json_file_path)
    
    if not vhdl_files:
        print("❌ No VHDL files loaded. Check your JSON file format.")
        return None
    
    # Check Ollama connection first
    if not await check_ollama_connection():
        print("❌ Cannot proceed without Ollama connection.")
        return None
    
    # Process all files in batches
    summary = await generator.process_all_vhdl_files_in_batches(vhdl_files, max_concurrent)
    
    return summary
# Full production run with your desired settings
async def run_full_processing():
    """Run the full processing with production settings"""
    print("🚀 Starting FULL PRODUCTION processing...")
    
    summary = await process_hdl_coder_json_in_batches(
        json_file_path="hdl_coder_20.json",
        max_generations_per_json=10,  # Your desired batch size
        max_concurrent=3              # Adjust based on your GPU capacity
    )
    
    return summary

# This is the standard way to run an async program from the top level.
if __name__ == "__main__":
    asyncio.run(run_full_processing())
