import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QuizEngine:
    
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.quizzes_path = Path("sessions/quizzes")
        self.quizzes_path.mkdir(parents=True, exist_ok=True)
    
    def generate_quiz(self, num_questions: int = 10, topic: Optional[str] = None) -> Dict:
        logger.info(f"Generating quiz with {num_questions} questions")
        
        topic_text = f" on the topic of {topic}" if topic else ""
        
        if topic:
            context_query = f"Explain {topic} in detail"
        else:
            context_query = "Provide an overview of the main topics covered"
        
        retriever = self.rag_engine.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(context_query)
        context = "\n\n".join([node.get_content() for node in nodes])
        
        prompt = f"""Based on the text below, generate {num_questions} multiple choice questions{topic_text}.

Text:
{context[:2500]}

Format:
Question 1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct: [Letter]
Explanation: [Text]

Question 2: [Question]
...

Generate exactly {num_questions} questions. Do not output anything else.

Question 1:"""
        
        response = self.rag_engine.llm.complete(prompt)
        quiz_text = "Question 1:" + response.text.strip()
        logger.info(f"Raw quiz text: {quiz_text[:500]}...")
        
        quiz = self._parse_quiz(quiz_text, num_questions)
        
        self._save_quiz(quiz)
        
        return quiz
    
    def _parse_quiz(self, quiz_text: str, expected_questions: int) -> Dict:
        questions = []
        
        question_blocks = re.split(r'Question\s*\d+:', quiz_text, flags=re.IGNORECASE)
        
        question_blocks = [b.strip() for b in question_blocks if b.strip()]
        
        for i, block in enumerate(question_blocks[:expected_questions], 1):
            try:
                lines = block.split('\n')
                question_text = lines[0].strip()
                
                options = {}
                correct_answer = "A"
                explanation = ""
                
                current_option = None
                
                for line in lines[1:]:
                    line = line.strip()
                    if not line: continue
                    
                    option_match = re.match(r'^([A-D])[\)\.]\s*(.+)', line, re.IGNORECASE)
                    if option_match:
                        current_option = option_match.group(1).upper()
                        options[current_option] = option_match.group(2).strip()
                    elif line.lower().startswith("correct:"):
                        correct_match = re.search(r'Correct:\s*([A-D])', line, re.IGNORECASE)
                        if correct_match:
                            correct_answer = correct_match.group(1).upper()
                    elif line.lower().startswith("explanation:"):
                        explanation = line.split(":", 1)[1].strip()
                    elif current_option:
                        options[current_option] += " " + line
                
                for letter in ['A', 'B', 'C', 'D']:
                    if letter not in options:
                        options[letter] = f"Option {letter}"
                
                correct_text = options.get(correct_answer, options['A'])
                
                questions.append({
                    "question": question_text,
                    "options": [options['A'], options['B'], options['C'], options['D']],
                    "correct_answer": correct_text,
                    "explanation": explanation
                })
                
            except Exception as e:
                logger.warning(f"Error parsing question block {i}: {e}")
                continue
        
        while len(questions) < expected_questions:
            questions.append({
                "question": f"Question {len(questions) + 1} (Generation failed)",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "explanation": "Could not generate a valid question."
            })
        
        return {
            "course": self.rag_engine.course_name,
            "num_questions": len(questions),
            "questions": questions
        }
    
    def _save_quiz(self, quiz: Dict):
        quiz_file = self.quizzes_path / f"quiz_{quiz['course']}.txt"
        
        with open(quiz_file, 'w', encoding='utf-8') as f:
            f.write(f"Quiz for Course: {quiz['course']}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, q in enumerate(quiz['questions'], 1):
                f.write(f"Q{i}: {q['question']}\n")
                for j, option in enumerate(q['options'], 1):
                    f.write(f"  {chr(64+j)}) {option}\n")
                f.write(f"Correct: {q['correct_answer']}\n")
                f.write(f"Explanation: {q['explanation']}\n\n")
    
    def score_quiz(self, quiz: Dict, user_answers: Dict[int, str]) -> Dict:
        correct = 0
        total = len(quiz['questions'])
        
        for i, question in enumerate(quiz['questions'], 1):
            correct_answer = question['correct_answer']
            user_answer = user_answers.get(i, "")
            
            if user_answer == correct_answer:
                correct += 1
        
        percentage = (correct / total * 100) if total > 0 else 0
        
        return {
            "correct": correct,
            "total": total,
            "percentage": percentage
        }