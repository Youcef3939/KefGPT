import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SessionManager:
    
    def __init__(self, history_file: str = "sessions/history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "qa_history": [],
            "quiz_results": []
        }
    
    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_qa(self, question: str, answer: str, course: Optional[str] = None):
        qa_entry = {
            "question": question,
            "answer": answer,
            "course": course,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history["qa_history"].append(qa_entry)
        self._save_history()
    
    def add_quiz_result(self, course: str, quiz: Dict, score: Dict):
        quiz_result = {
            "course": course,
            "quiz_id": f"quiz_{course}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "num_questions": quiz.get("num_questions", 0),
            "score": score,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history["quiz_results"].append(quiz_result)
        self._save_history()
    
    def get_history(self) -> Dict:
        return self.history
    
    def get_qa_history(self, course: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        qa_history = self.history["qa_history"]
        
        if course:
            qa_history = [qa for qa in qa_history if qa.get("course") == course]
        
        if limit:
            qa_history = qa_history[-limit:]
        
        return qa_history
    
    def get_quiz_results(self, course: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        quiz_results = self.history["quiz_results"]
        
        if course:
            quiz_results = [qr for qr in quiz_results if qr.get("course") == course]
        
        if limit:
            quiz_results = quiz_results[-limit:]
        
        return quiz_results