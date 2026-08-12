import requests
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from src.utils import setup_logger, ensure_directory_exists, save_json
from src.config import Config
import re

logger = setup_logger(__name__, "logs/calculator.log")


class Calculator:
    """
    Class responsible for calculating variables based on formulas using Qwen-Max API
    """

    def __init__(self):
        self.api_key = Config.QWEN_MAX_API_KEY
        # Using the correct endpoint for qwen-max (text-based model) according to DashScope API
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.result_output_dir = Config.RESULT_OUTPUT_DIR

    def calculate_variables(self, pdf_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate variables based on formulas using Qwen-Max API
        """
        if not self.api_key:
            logger.error("QWEN_MAX_API_KEY not set in environment variables")
            return {}

        all_variables_collected = input_data.get("all_variables_collected", {})
        calculation_formulas = input_data.get("calculation_formulas", {})

        results = {
            "pdf_name": pdf_name,
            "calculated_variables": {},
            "calculation_metadata": {},
            "missing_info_summary": {}  # Track what info was missing for each indicator
        }

        for indicator_name, formulas in calculation_formulas.items():
            if isinstance(formulas, list) and len(formulas) > 0:
                variables_for_indicator = all_variables_collected.get(indicator_name, {})

                # Prepare context for the LLM
                context = {
                    "indicator": indicator_name,
                    "available_variables": variables_for_indicator,
                    "formulas": formulas
                }

                # Call the API to calculate variables based on formulas
                calculated_data = self.call_qwen_max_for_calculation(context)

                if calculated_data:
                    # Update results with calculated data
                    results["calculated_variables"][indicator_name] = calculated_data.get("calculated_variables", {})

                    # Extract metadata
                    metadata = calculated_data.get("metadata", {})
                    results["calculation_metadata"][indicator_name] = metadata

                    # Extract and summarize missing info for this indicator
                    missing_info = metadata.get("missing_info", [])
                    if missing_info:
                        results["missing_info_summary"][indicator_name] = missing_info

        # Create subdirectory for this PDF's calculation results
        pdf_result_dir = self.result_output_dir / pdf_name
        ensure_directory_exists(pdf_result_dir)

        # Save the calculation results
        output_file = pdf_result_dir / f"{pdf_name}_calculation_result.json"
        save_json(results, output_file)

        logger.info(f"Saved calculation results for {pdf_name} to {output_file}")

        return results

    def call_qwen_max_for_calculation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Qwen-Max API to calculate variables based on formulas
        """
        if not self.api_key:
            logger.error("QWEN_MAX_API_KEY not set in environment variables")
            return {}

        # Prepare the prompt with proper context
        indicator = context['indicator']
        available_vars = context['available_variables']
        formulas = context['formulas']

        # Format the prompt to be very specific about the input data and required output
        prompt = (
            "You are an expert calculator for ESG metrics. I will provide:\n"
            f"1. Indicator: {indicator}\n"
            f"2. Available variables: {json.dumps(available_vars, ensure_ascii=False, indent=2)}\n"
            f"3. Formulas to apply: {json.dumps(formulas, ensure_ascii=False, indent=2)}\n\n"

            "Based on the available variables and the formulas, calculate the requested values.\n\n"

            "For each formula, check if you have all required information:\n"
            "- If you have all required data to compute the formula, calculate it and return the result\n"
            "- If you're missing required information to compute a formula, return '信息缺失：信息' for that calculation\n\n"

            "Return your response in strict JSON format with the following structure:\n"
            "{\n"
            "  \"calculated_variables\": {\n"
            "    \"formula_explanation\": {\n"
            "      \"formula\": \"actual_formula_text\",\n"
            "      \"result\": calculated_value_or_null,\n"
            "      \"status\": \"calculated\" or \"information_missing\",\n"
            "      \"reason\": \"explanation_of_calculation_or_reason_for_missing_info\"\n"
            "    }\n"
            "  },\n"
            "  \"metadata\": {\n"
            "    \"input_variables\": [list_of_all_available_variables],\n"
            "    \"used_variables\": [list_of_variables_actually_used],\n"
            "    \"missing_info\": [list_of_information_needed_but_not_available]\n"
            "  }\n"
            "}\n\n"

            "IMPORTANT: Respond ONLY with valid JSON in the specified format. Do not add any explanatory text outside the JSON."
        )

        # Prepare the payload - using the correct format for DashScope text generation API
        # 与 llm_matching 统一用 QWEN_MAX_MODEL（默认 qwen3.7-max），2026-08-03 起不再硬编码 qwen-max
        # 2026-08-13: add enable_thinking support for qwen3.8-max reasoning mode
        params = {
            "max_tokens": Config.THINKING_MAX_TOKENS if Config.ENABLE_THINKING else 4000,
            "temperature": 0.1,
            "result_format": "message",
        }
        if Config.ENABLE_THINKING:
            params["enable_thinking"] = True
            logger.info(f"[Calculator] Thinking enabled (model={Config.QWEN_MAX_MODEL}, max_tokens={params['max_tokens']})")

        payload = {
            "model": Config.QWEN_MAX_MODEL,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": params,
        }

        try:
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=(10, 600),  # connect 10s, read 600s (thinking mode is slow)
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"[Calculator] API attempt {attempt+1} failed: {e}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"[Calculator] API failed after {max_retries} attempts: {e}")
                        raise

            result = response.json()

            # Extract the text content from the response using the standard DashScope response format
            text_content = ""

            # Check for the standard response structure
            if 'output' in result and 'choices' in result['output']:
                choices = result['output']['choices']
                if len(choices) > 0 and 'message' in choices[0]:
                    message = choices[0]['message']

                    # Log reasoning_content if present (thinking mode)
                    reasoning = message.get('reasoning_content', '')
                    if reasoning:
                        logger.debug(f"[Calculator] Reasoning (truncated): {reasoning[:200]}...")

                    if 'content' in message:
                        content = message['content']
                        # Handle both string and list content formats
                        if isinstance(content, str):
                            text_content = content
                        elif isinstance(content, list):
                            # For list content, extract text parts
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                            text_content = ' '.join(text_parts)

                # If content still not found, try alternative structures
                if not text_content:
                    if 'output' in result and 'text' in result['output']:
                        text_content = result['output']['text']
                    elif 'result' in result:
                        text_content = result['result']
                    else:
                        # Log the full response for debugging
                        logger.debug(f"Unexpected response format: {result}")
                        text_content = str(result)

                # Attempt to extract JSON from the response
                # Look for JSON within triple backticks if present
                import re
                # First check for code blocks
                code_block_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text_content, re.DOTALL)
                if code_block_match:
                    json_text = code_block_match.group(1)
                else:
                    # Find JSON between first { and last }
                    brace_start = text_content.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        for i, char in enumerate(text_content[brace_start:], brace_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_text = text_content[brace_start:i+1]
                                    break
                        else:
                            json_text = ""  # No matching braces found
                    else:
                        json_text = ""

                # Try to parse the extracted JSON
                if json_text:
                    try:
                        parsed_result = json.loads(json_text)
                        return parsed_result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse extracted JSON: {e}")
                        logger.debug(f"Extracted text: {json_text[:500]}...")

                # If JSON parsing failed, return default structure with raw response
                logger.info("Could not parse JSON from API response, returning default structure")
                return {
                    "calculated_variables": {},
                    "metadata": {
                        "input_variables": list(available_vars.keys()) if isinstance(available_vars, dict) else [],
                        "used_variables": [],
                        "missing_info": ["Could not parse response from model - no valid JSON found"],
                        "raw_response": text_content[:1000] if text_content else "No content received"
                    }
                }

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error calling Qwen-Max API: {str(e)}")
            if response.status_code == 401:
                logger.error("Authentication failed - please check your QWEN_MAX_API_KEY")
            elif response.status_code == 403:
                logger.error("Access forbidden - please check API key permissions")
            elif response.status_code == 404:
                logger.error("API endpoint not found - please check the URL")
            elif response.status_code >= 400:
                try:
                    error_detail = response.json()
                    logger.error(f"API Error: {error_detail}")
                except:
                    logger.error(f"API Error (text): {response.text}")
            return {"error": f"HTTP {response.status_code}: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Error calling Qwen-Max API: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {str(e)}")
            return {
                "error": f"JSON decode error: {str(e)}",
                "metadata": {
                    "input_variables": list(available_vars.keys()) if isinstance(available_vars, dict) else [],
                    "used_variables": [],
                    "missing_info": ["Could not parse model response - invalid JSON"]
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            return {
                "error": f"Unexpected error: {str(e)}",
                "metadata": {
                    "input_variables": list(available_vars.keys()) if isinstance(available_vars, dict) else [],
                    "used_variables": [],
                    "missing_info": ["Unexpected error occurred during processing"]
                }
            }

    def process_calculation_for_specific_pdfs(self, pdf_names: List[str]) -> Dict[str, Any]:
        """
        Process calculation for specific PDFs by name
        """
        if not self.api_key:
            logger.error("QWEN_MAX_API_KEY not set in environment variables")
            return {}

        all_results = {}

        for pdf_name in pdf_names:
            # Look for the quantitative_analysis.json file for this specific PDF
            pdf_quantitative_dir = Config.QUANTITATIVE_RESULT_DIR / pdf_name
            analysis_file = pdf_quantitative_dir / f"{pdf_name}_quantitative_analysis.json"

            if not analysis_file.exists():
                logger.warning(f"Quantitative analysis file not found for {pdf_name}: {analysis_file}")
                continue

            try:
                logger.info(f"Processing calculation for: {pdf_name}")

                # Load the quantitative analysis file
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)

                # Perform calculations
                result = self.calculate_variables(pdf_name, input_data)
                all_results[pdf_name] = result
            except Exception as e:
                logger.error(f"Failed to process calculation for {pdf_name}: {str(e)}")
                continue

        logger.info(f"Calculation processing completed for {len(all_results)} PDFs")

        return all_results

    def process_calculation_for_all_pdfs(self) -> Dict[str, Any]:
        """
        Process calculation for all PDFs in the quantitative result directory
        """
        if not self.api_key:
            logger.error("QWEN_MAX_API_KEY not set in environment variables")
            return {}

        pdf_folders = []
        if Config.QUANTITATIVE_RESULT_DIR.exists():
            for item in Config.QUANTITATIVE_RESULT_DIR.iterdir():
                if item.is_dir():
                    pdf_name = item.name

                    # Look for the quantitative_analysis.json file in this folder
                    analysis_file = item / f"{pdf_name}_quantitative_analysis.json"
                    if analysis_file.exists():
                        pdf_folders.append((item, pdf_name))

        if not pdf_folders:
            logger.warning(f"No PDF folders with quantitative_analysis.json found in {Config.QUANTITATIVE_RESULT_DIR}")
            return {}

        logger.info(f"Found {len(pdf_folders)} PDF folders to process for calculation")

        all_results = {}

        for i, (pdf_folder, pdf_name) in enumerate(pdf_folders, 1):
            try:
                logger.info(f"Processing calculation for {i}/{len(pdf_folders)}: {pdf_name}")

                # Load the quantitative analysis file
                analysis_file = pdf_folder / f"{pdf_name}_quantitative_analysis.json"
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)

                # Perform calculations
                result = self.calculate_variables(pdf_name, input_data)
                all_results[pdf_name] = result
            except Exception as e:
                logger.error(f"Failed to process calculation for {pdf_name}: {str(e)}")
                continue

        logger.info("Calculation processing completed")

        return all_results