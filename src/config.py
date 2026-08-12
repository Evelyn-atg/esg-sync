import os
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration class to manage application settings"""

    # API Keys
    QWEN_VL_PLUS_API_KEY = os.getenv('QWEN_VL_PLUS_API_KEY')
    QWEN_MAX_API_KEY = os.getenv('QWEN_MAX_API_KEY')
    QWEN_MAX_MODEL = os.getenv('QWEN_MAX_MODEL', 'qwen3.7-max')
    DATALAB_API_KEY = os.getenv('DATALAB_API_KEY')

    # === Qwen3.8 Reasoning Mode (optional, off by default) ===
    # Enable deep thinking for complex indicator transformations.
    # Set ENABLE_THINKING=true and QWEN_MAX_MODEL=qwen3.8-max in .env or shell.
    ENABLE_THINKING = os.getenv('ENABLE_THINKING', 'false').lower() in ('1', 'true', 'yes')
    # Thinking budget: max tokens for reasoning_content (the CoT trace).
    # When the budget is reached, the model stops thinking and outputs the answer.
    # Recommended: 8192+ for complex tasks (ESG indicator matching with 90+ records).
    THINKING_BUDGET = int(os.getenv('THINKING_BUDGET', '8192'))
    # Max tokens for TOTAL output (reasoning_content + final answer combined).
    # Per DashScope docs: max_tokens = thinking_budget + answer_budget.
    # Qwen3.8-max supports up to 131K output in thinking mode; 262K reasoning.
    # 16384 = 8192 (thinking) + 8192 (JSON answer, enough for 90+ records).
    THINKING_MAX_TOKENS = int(os.getenv('THINKING_MAX_TOKENS', '16384'))
    # VL model name (qwen3.8-max is multimodal; legacy default: qwen-vl-plus).
    QWEN_VL_MODEL = os.getenv('QWEN_VL_MODEL', 'qwen-vl-plus')

    # Directory configurations
    PDF_INPUT_DIR = Path(os.getenv('PDF_INPUT_DIR', './HKEX ESG Reports'))
    TEXT_OUTPUT_DIR = Path(os.getenv('TEXT_OUTPUT_DIR', './extracted_text'))
    IMAGE_OUTPUT_DIR = Path(os.getenv('IMAGE_OUTPUT_DIR', './page_images'))
    PARAGRAPH_OUTPUT_DIR = Path(os.getenv('PARAGRAPH_OUTPUT_DIR', './paragraphs'))
    INDEX_OUTPUT_DIR = Path(os.getenv('INDEX_OUTPUT_DIR', './search_index'))
    UNSTRUCTURED_RESULT_DIR = Path(os.getenv('UNSTRUCTURED_RESULT_DIR', './keyword_match_results'))
    QUANTITATIVE_RESULT_DIR = Path(os.getenv('QUANTITATIVE_RESULT_DIR', './quantitative_results_Qwen'))
    CHANDRA_OCR_RESULT_DIR = Path(os.getenv('CHANDRA_OCR_RESULT_DIR', './quantitative_results_ocr/chandra_ocr_2'))
    NUMERIC_EXTRACT_DIR = Path(os.getenv('NUMERIC_EXTRACT_DIR', './numeric_extracts'))
    QUANTITATIVE_VARIABLES_FILE = Path(
        os.getenv('QUANTITATIVE_VARIABLES_FILE', './quantitative_variables.json')
    )
    QUANTITATIVE_VARIABLE_DEFINITIONS_FILE = Path(
        os.getenv('QUANTITATIVE_VARIABLE_DEFINITIONS_FILE', './quantitative_variable_definitions.json')
    )

    # New result directory for calculation results
    RESULT_OUTPUT_DIR = Path(os.getenv('RESULT_OUTPUT_DIR', './calculation_results'))

    # Log directory
    LOG_DIR = Path(os.getenv('LOG_DIR', './logs'))

    # Processing configurations
    KEYWORD_MATCHING_MAX_WORKERS = int(os.getenv('KEYWORD_MATCHING_MAX_WORKERS', 1))
    LLM_MATCHING_MAX_WORKERS = int(os.getenv('LLM_MATCHING_MAX_WORKERS', 4))
    MAX_PAGES_FOR_IMAGE_RECOGNITION = int(os.getenv('MAX_PAGES_FOR_IMAGE_RECOGNITION', 20))
    IMAGE_RECOGNITION_MAX_WORKERS = int(os.getenv('IMAGE_RECOGNITION_MAX_WORKERS', 3))
    CLEANUP_IMAGES_AFTER_RECOGNITION = os.getenv('CLEANUP_IMAGES_AFTER_RECOGNITION', 'true').lower() in ('1', 'true', 'yes')
    # OCR backend: 'chandra_local' (local GPU model via chandra-ocr) or 'datalab_api' (cloud Convert API)
    OCR_BACKEND = os.getenv('OCR_BACKEND', 'chandra_local')
    # Local path to Chandra OCR model (downloaded from ModelScope, not HuggingFace)
    CHANDRA_MODEL_PATH = os.getenv('CHANDRA_MODEL_PATH', os.path.expanduser('~/models/chandra-ocr-2'))
    # Qwen backend for quantitative normalization: 'dashscope_api' (cloud) or 'local' (local Qwen2.5-VL on GPU)
    QWEN_BACKEND = os.getenv('QWEN_BACKEND', 'dashscope_api')
    QWEN_VL_LOCAL_CHECKPOINT = os.getenv('QWEN_VL_LOCAL_CHECKPOINT', os.path.expanduser('~/models/qwen2.5-vl-7b'))
    # When set, skip the slow Chandra OCR step and reuse each PDF's cached
    # <pdf>_ocr_output.json, re-running only the Qwen/matching/aggregation downstream
    # (text-only, from the cached table HTML). Use for fast iteration on extraction logic.
    REUSE_CACHED_OCR = os.getenv('REUSE_CACHED_OCR', '0') not in ('0', '', 'false', 'False')
    # When set, log each table's raw Qwen response before normalization (diagnose
    # tables the 7B emits nothing for, e.g. 1345). Verbose — for debugging only.
    DEBUG_EXTRACTION = os.getenv('DEBUG_EXTRACTION', '0') not in ('0', '', 'false', 'False')
    # When table extraction finds NO scope emissions, read candidate pages as full
    # page images (recovers numbers that live in infographics/callouts, not tables).
    VISION_SCOPE_FALLBACK = os.getenv('VISION_SCOPE_FALLBACK', '1') not in ('0', '', 'false', 'False')
    DATALAB_CONVERT_MODE = os.getenv('DATALAB_CONVERT_MODE', 'accurate')
    DATALAB_API_MAX_POLLS = int(os.getenv('DATALAB_API_MAX_POLLS', 300))
    DATALAB_API_POLL_INTERVAL = int(os.getenv('DATALAB_API_POLL_INTERVAL', 1))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def initialize_directories(cls):
        """Create necessary directories if they don't exist"""
        dirs = [
            cls.TEXT_OUTPUT_DIR,
            cls.IMAGE_OUTPUT_DIR,
            cls.PARAGRAPH_OUTPUT_DIR,
            cls.INDEX_OUTPUT_DIR,
            cls.UNSTRUCTURED_RESULT_DIR,
            cls.NUMERIC_EXTRACT_DIR,
            cls.QUANTITATIVE_RESULT_DIR,  # 新增定量结果目录
            cls.CHANDRA_OCR_RESULT_DIR,  # Chandra OCR 独立测试结果目录
            cls.RESULT_OUTPUT_DIR,  # 新增计算结果目录
            cls.LOG_DIR,
        ]

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def setup_logging(cls):
        """Setup basic logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_DIR / 'app.log'),
                logging.StreamHandler()
            ]
        )
