# KB-AnswerScorer

A tool for evaluating LLM responses against a knowledge base of expert solutions.

## Overview

KB-AnswerScorer is a utility designed to evaluate how well large language models (LLMs) answer customer support questions compared to established expert solutions. The tool processes questions, obtains model responses via OpenWebUI, compares these responses to reference solutions using semantic and lexical similarity metrics, and generates comprehensive reports.

The comparison is based on:

- BERTScore (semantic similarity)
- F1 Score (lexical similarity)
- BLEU Score (translation quality metric)

## Installation

Clone this repository:

```bash
git clone https://github.com/farithadnan/KB-AnswerScorer.git
cd KB-AnswerScorer
```

Create virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# cmd
path\to\venv\Scripts\activate

# powershell
.\path\to\venv\Scripts\Activate

# bash
source path/to/venv/bin/activate
```

## Project Structure

```yaml
KB-AnswerScorer/
├── main.py                  # Main script
├── .env                     # Environment variables
├── data/                    # Input data directory
│   ├── questions.xlsx       # Customer questions
│   └── solutions.xlsx       # Expert solutions
├── metrics/                 # Scoring modules
│   ├── score_calculator.py  # Implements scoring metrics
│   └── solution_matcher.py  # Matches responses to solutions
├── opwebui/                 # OpenWebUI integration
│   └── api_client.py        # Client for OpenWebUI API
└── utils/                   # Utility modules
    ├── data_extractor.py    # Parses Excel input files
    ├── quality_filter.py    # Assesses response quality
    └── report_generator.py  # Generates evaluation reports
```

## Configuration

Create a `.env` file in the project root with the following variables:

```yaml
# Directory and file locations
DATA_DIR_PATH=./data
QUESTION_EXCEL=questions.xlsx
SOLUTION_EXCEL=solutions.xlsx
QUESTION_SHEET_NAME=Questions

# OpenWebUI Configuration
OPENWEBUI_API_URL=http://your-openwebui-instance:5000/api/chat
OPENWEBUI_JWT_TOKEN=your_jwt_token_here
```

### Getting the OpenWebUI JWT Token

> Check the [official guide](https://docs.openwebui.com/getting-started/api-endpoints#authentication) for more details about this.

1. Login to your OpenWebUI instance
2. Open the browser developer tools (F12)
3. Go to the Application tab
4. Look for "Local Storage" and find the entry for your OpenWebUI domain
5. Copy the JWT token value (it starts with "ey...")


### Input Files

The tool expects two Excel files:

1. **Question file**: Contains customer queries/issues.
    - **Required structure:**
        - Excel file with customer questions (header starts at row 3)
        - Main question text in column B
        - Solutions used in column C (optional)
        - AI Solutions used in column D (optional)
        - Each row represents a unique customer issue
    - **Parsed into:**
        - `id`: Automatically assigned based on row number (starting from 1)
        - `issue`: The customer question text from column B
        - `solutions_used`: List of solution indices that can be manually set later
        - `ai_solutions_used`: List of AI solution indices from column D
2. **Solutions file**: Contains expert solutions.
    - **Required structure:**
        - Excel file with solutions (header in row 1)
        - Column A: Solution text with title and steps
        - Column B: Optional error messages
    - **Parsed into:**
        - `id`: Automatically assigned based on row number (starting from 1)
        - `title`: Extracted from the first line of the solution text (e.g., "Solutions 1")
        - `steps`: Each line after the title becomes a step in the solution
        - `error_message`: Any text in column B becomes the error message

You can check `data/samples` to see the sample of these two files.

## Usage

Run the script with various command-line options to control its behavior:

```bash
python main.py [options]
```

Command line options:

| Options            | Descriptions                              | Default           |
|--------------------|-------------------------------------------|-------------------|
|`--limit N`         | Process only the first N questions        | 0 (all questions) |
|`--question-id ID`  | Process only a specific question by ID	 | None              |
|`--verbose`, `-v`   | Display detailed logs and results	     | False             |
| `--report-dir DIR` | Directory to save reports                 | "reports"         |
| `--wait-time SEC`  | Wait time between API calls in seconds    | 1.0               |
| `--skip-report`    | Skip report generation                    | False             |

### Example

Process all questions:

```bash
python main.py
```

Process only the first 5 questions:

```bash
python main.py --limit 5
```

Process only specific question by ID:

```bash
python main.py --question-id 2
```

Generate detailed output for each questions:

```bash
python main.py --verbose
```

Save reports to a custom directory:

```bash
python main.py --report-dir my_reports
```

Adjust wait time between API calls:

```bash
python main.py --wait-time 2.0
```

## Troubleshooting

### No response from OpenWebUI

- Verify your JWT token is valid and not expired
- Check that the OpenWebUI API URL is correct
- Ensure OpenWebUI is running and accessible

### Missing files error

- Verify that the data directory and Excel files exist
- Check the paths in your .env file

### Low scores across all questions

- The model may not be suitable for your domain
- Consider adjusting the quality thresholds
- Review your reference solutions for clarity

### Environment Variable Issues

If you update your `.env` file and changes aren't detected:

- Make sure to use `#` for comments
- Restart your terminal/command prompt


### Excel Format Issues

If you're getting parsing errors:

- Check that the header row is set correctly (defaults to row 3)
- Verify column mappings in the DataExtractor configuration
- For solutions, ensure they follow the "Solution X" format with numbered steps

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.