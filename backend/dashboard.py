
"""
Dashboard-related API endpoints for the automation testing system.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

# Create router for dashboard endpoints
router = APIRouter()

@router.get("/reports")
async def get_test_results():
    """Serve all automation reports from timestamped report directories"""
    try:
        automation_dir = Path(__file__).parent.parent / "automation_testing"
        reports_dir = automation_dir / "reports"
        
        if not reports_dir.exists():
            return {"error": "Reports directory not found", "message": "Run automation tests first", "test_runs": []}
        
        test_runs = []
        
        # Scan all test run directories
        for test_dir in sorted(reports_dir.iterdir(), reverse=True):  # Most recent first
            if not test_dir.is_dir():
                continue
                
            report_path = test_dir / "report.json"
            preds_path = test_dir / "preds.jsonl"
            gold_path = test_dir / "gold.jsonl"
            
            if not report_path.exists():
                continue  # Skip directories without report files
            
            try:
                # Load main report data
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                
                test_run = {
                    'run_id': test_dir.name,
                    'summary': report_data.get('summary', {}),
                    'total_questions': len(report_data.get('per_question', [])),
                    'files_present': {
                        'report': report_path.exists(),
                        'predictions': preds_path.exists(),
                        'gold': gold_path.exists()
                    }
                }
                
                # Add detailed predictions if requested
                predictions_data = None
                if preds_path.exists() and gold_path.exists():
                    try:
                        # Load predictions
                        predictions = []
                        with open(preds_path, 'r') as f:
                            for line in f:
                                predictions.append(json.loads(line.strip()))
                        
                        # Load gold standard
                        gold_data = {}
                        with open(gold_path, 'r') as f:
                            for line in f:
                                item = json.loads(line.strip())
                                gold_data[item['id']] = item
                        
                        # Combine predictions with gold standard and per-question metrics
                        combined_predictions = []
                        for pred in predictions:
                            pred_id = pred['id']
                            if pred_id in gold_data:
                                gold_item = gold_data[pred_id]
                                
                                # Find matching per-question metrics from report
                                per_question_metrics = next(
                                    (m for m in report_data.get("per_question", []) if m["id"] == pred_id), 
                                    {}
                                )
                                
                                combined_predictions.append({
                                    'id': pred_id,
                                    'category': pred_id.split(':')[0],
                                    'query': gold_item['query'],
                                    'model_answer': pred['model_answer'],
                                    'reference_answer': gold_item['reference_answer'],
                                    'nuggets': gold_item['nuggets'],
                                    'retrieved_ids': pred['retrieved_ids'],
                                    'gold_passages': gold_item['gold_passages'],
                                    'url': gold_item['url'],
                                    'metrics': {
                                        'nugget_precision': per_question_metrics.get('nugget_precision', 0),
                                        'nugget_recall': per_question_metrics.get('nugget_recall', 0),
                                        'nugget_f1': per_question_metrics.get('nugget_f1', 0),
                                        'sbert_cosine': per_question_metrics.get('sbert_cosine', 0),
                                        'sbert_cosine_chunk': per_question_metrics.get('sbert_cosine_chunk', 0),
                                        'bertscore_f1': per_question_metrics.get('bertscore_f1', 0),
                                        'recall@1': per_question_metrics.get('recall@1', 0),
                                        'recall@3': per_question_metrics.get('recall@3', 0),
                                        'recall@5': per_question_metrics.get('recall@5', 0),
                                        'ndcg@1': per_question_metrics.get('ndcg@1', 0),
                                        'ndcg@3': per_question_metrics.get('ndcg@3', 0),
                                        'ndcg@5': per_question_metrics.get('ndcg@5', 0)
                                    }
                                })
                        
                        # Add predictions metadata
                        pred_stat_info = os.stat(preds_path)
                        # Dynamically calculate categories from actual data
                        categories = {}
                        for pred in combined_predictions:
                            category = pred['category']
                            categories[category] = categories.get(category, 0) + 1
                        predictions_data = {
                            'predictions': combined_predictions,
                            'total_questions': len(combined_predictions),
                            'categories': categories,
                            'predictions_timestamp': datetime.fromtimestamp(pred_stat_info.st_mtime).isoformat()
                        }
                    except Exception as pred_error:
                        print(f"Error loading predictions for {test_dir.name}: {pred_error}")
                        # Continue without predictions if there's an error
                
                # Add predictions data to test run
                if predictions_data:
                    test_run['predictions_data'] = predictions_data
                    
                test_runs.append(test_run)
                
            except Exception as e:
                print(f"Error loading test run {test_dir.name}: {e}")
                continue
        
        return {
            'test_runs': test_runs,
            'total_runs': len(test_runs),
            'latest_run': test_runs[0] if test_runs else None
        }
    except Exception as e:
        return {"error": "Failed to load test results", "message": str(e), "test_runs": []}

@router.post("/run-tests")
async def run_tests():
    """Run the automation testing suite and return status"""
    try:
        # Path to the run_tests.py script
        automation_dir = Path(__file__).parent.parent / "automation_testing"
        run_tests_script = automation_dir / "run_tests.py"
        
        if not run_tests_script.exists():
            return {"error": "Test runner not found", "message": "run_tests.py script is missing"}
        
        # Run the test script
        result = subprocess.run(
            [sys.executable, str(run_tests_script)],
            cwd=str(automation_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Tests completed successfully",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            return {
                "status": "error", 
                "message": "Tests failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {"error": "Test execution timeout", "message": "Tests took longer than 5 minutes to complete"}
    except Exception as e:
        return {"error": "Failed to run tests", "message": str(e)}

@router.get("/dashboard")
async def serve_dashboard():
    """Serve the dashboard HTML file"""
    dashboard_path = Path(__file__).parent.parent / "frontend" / "out" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    else:
        return {"error": "Dashboard not found", "message": "Build the frontend first"}
