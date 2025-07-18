"""
Evaluation module for the FAQ agent
"""
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from pdfminer.high_level import extract_text
import os
from deployed_agent.graph import graph, evaluate_agent, AgentState
from langchain_core.messages import HumanMessage
from rouge_score import rouge_scorer
import time
import sys
from threading import Thread
from itertools import cycle

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

class ProgressAnimation:
    """Simple progress animation class"""
    def __init__(self, message="Processing"):
        self.message = message
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the animation"""
        self.running = True
        self.thread = Thread(target=self._animate)
        self.thread.start()
        
    def stop(self):
        """Stop the animation"""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        sys.stdout.write('\r' + ' ' * 50 + '\r')
        sys.stdout.flush()
        
    def _animate(self):
        """Animation loop"""
        spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        while self.running:
            sys.stdout.write(f'\r{next(spinner)} {self.message}...')
            sys.stdout.flush()
            time.sleep(0.1)

def print_progress_bar(current, total, length=50):
    """Print a progress bar"""
    filled = int(length * current / total)
    bar = '█' * filled + '-' * (length - filled)
    percent = (current / total) * 100
    sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({current}/{total})')
    sys.stdout.flush()

def evaluate_response(generated_answer: str, ground_truth: str) -> Dict[str, float]:
    """
    Evaluate the generated answer against ground truth using ROUGE scores
    """
    scores = scorer.score(ground_truth, generated_answer)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def evaluate_single_query(query: str, ground_truth: str, query_num: int = 0, total_queries: int = 0, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the agent with a query and evaluate its response against ground truth
    """
    # Show progress for current query
    if total_queries > 0:
        print(f"\n🔍 Evaluating query {query_num}/{total_queries}")
        if category:
            print(f"Category: {category}")
        print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    # Start animation for agent processing
    animation = ProgressAnimation("Running FAQ agent")
    animation.start()
    
    try:
        # Use the evaluate_agent function from the deployed agent
        result = evaluate_agent(query, ground_truth)
        
        # Stop animation
        animation.stop()
        
        # Show completion
        print(f"✅ Query {query_num} completed")
        print(f"   Search attempts: {result.get('search_attempts', 'N/A')}")
        print(f"   Answer found: {result.get('answer_found', 'N/A')}")
        print(f"   ROUGE-L score: {result.get('rouge_scores', {}).get('rougeL', 'N/A'):.4f}")
        
        # Add category to result
        if category:
            result['category'] = category
        
        return result
        
    except Exception as e:
        animation.stop()
        print(f"❌ Error evaluating query {query_num}: {str(e)}")
        return {
            "query": query,
            "generated_answer": f"Error: {str(e)}",
            "ground_truth": ground_truth,
            "rouge_scores": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "search_attempts": 0,
            "answer_found": False,
            "category": category
        }

def extract_qa_pairs_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extract question-answer pairs from the FAQ PDF document using doc_functions
    """
    print("📄 Extracting Q&A pairs from PDF...")
    animation = ProgressAnimation("Parsing PDF document")
    animation.start()
    
    try:
        # Import the doc functions
        from docs.doc_functions import doc_extractor, doc_mapper, create_question_chunks
        
        # Extract text from PDF
        extracted_text = doc_extractor()
        animation.stop()
        
        print("✅ PDF extracted successfully")
        print("🔍 Parsing Q&A pairs...")
        
        # Map document content by headers and parse Q&A pairs
        animation = ProgressAnimation("Mapping document structure")
        animation.start()
        
        doc_map = doc_mapper(extracted_text)
        animation.stop()
        
        print("✅ Document structure mapped")
        print("📝 Creating question chunks...")
        
        # Create question chunks
        animation = ProgressAnimation("Creating question chunks")
        animation.start()
        
        chunks = create_question_chunks(doc_map)
        animation.stop()
        
        # Convert chunks to the format expected by the evaluation
        qa_pairs = []
        for chunk in chunks:
            if chunk['question'] and chunk['answer']:
                qa_pairs.append({
                    "question": chunk['question'],
                    "answer": chunk['answer'],
                    "category": chunk['category']
                })
        
        print(f"✅ Found {len(qa_pairs)} Q&A pairs across {len(doc_map)} categories")
        
        # Show category breakdown
        category_counts = {}
        for pair in qa_pairs:
            category = pair.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print("📊 Category breakdown:")
        for category, count in category_counts.items():
            print(f"   • {category}: {count} questions")
        
        return qa_pairs
        
    except Exception as e:
        animation.stop()
        print(f"❌ Error extracting Q&A pairs: {str(e)}")
        return []

def run_evaluation():
    """
    Run evaluation on the FAQ agent
    """
    print("🚀 Starting FAQ Agent Evaluation")
    print("=" * 40)
    
    # Path to FAQ document
    faq_path = os.path.join('docs', 'FAQ Dexa Medica.pdf')
    
    if not os.path.exists(faq_path):
        print(f"❌ FAQ document not found at: {faq_path}")
        return None
    
    # Extract QA pairs
    qa_pairs = extract_qa_pairs_from_pdf(faq_path)
    
    if not qa_pairs:
        print("❌ No Q&A pairs found in the document")
        return None
    
    print(f"\n📊 Starting evaluation of {len(qa_pairs)} queries...")
    print("=" * 40)
    
    # Run evaluation for each pair
    results = []
    total_queries = len(qa_pairs)
    
    for i, pair in enumerate(qa_pairs, 1):
        result = evaluate_single_query(
            pair["question"], 
            pair["answer"], 
            i, 
            total_queries,
            pair.get("category")
        )
        results.append(result)
        
        # Show overall progress
        print_progress_bar(i, total_queries)
        time.sleep(0.1)  # Small delay to show progress
    
    print(f"\n\n🎉 Evaluation completed!")
    print("=" * 40)
    
    # Process results
    print("📈 Calculating metrics...")
    animation = ProgressAnimation("Processing results")
    animation.start()
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Calculate average scores
    avg_scores = {
        'rouge1': df['rouge_scores'].apply(lambda x: x['rouge1']).mean(),
        'rouge2': df['rouge_scores'].apply(lambda x: x['rouge2']).mean(),
        'rougeL': df['rouge_scores'].apply(lambda x: x['rougeL']).mean()
    }
    
    # Calculate category-based metrics
    category_metrics = {}
    if 'category' in df.columns:
        for category in df['category'].unique():
            if pd.notna(category):
                category_df = df[df['category'] == category]
                category_metrics[category] = {
                    'total_queries': len(category_df),
                    'avg_search_attempts': category_df['search_attempts'].mean(),
                    'answers_found_rate': category_df['answer_found'].mean(),
                    'rouge_scores': {
                        'rouge1': category_df['rouge_scores'].apply(lambda x: x['rouge1'] if isinstance(x, dict) else 0.0).mean(),
                        'rouge2': category_df['rouge_scores'].apply(lambda x: x['rouge2'] if isinstance(x, dict) else 0.0).mean(),
                        'rougeL': category_df['rouge_scores'].apply(lambda x: x['rougeL'] if isinstance(x, dict) else 0.0).mean()
                    }
                }
    
    # Calculate other metrics
    metrics = {
        'total_queries': len(df),
        'avg_search_attempts': df['search_attempts'].mean(),
        'answers_found_rate': df['answer_found'].mean(),
        'rouge_scores': avg_scores,
        'category_metrics': category_metrics
    }
    
    animation.stop()
    
    # Save results
    print("💾 Saving results...")
    df.to_csv('evaluation_results.csv', index=False)
    
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✅ Results saved to evaluation_results.csv and evaluation_metrics.json")
    
    return metrics

if __name__ == "__main__":
    # Make sure LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set
    if not os.getenv("LANGCHAIN_TRACING_V2") or not os.getenv("LANGSMITH_API_KEY"):
        print("⚠️  Please set LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY environment variables")
        exit(1)
    
    print("🔧 Environment variables configured")
    
    metrics = run_evaluation()
    
    if metrics:
        print("\n📊 Final Evaluation Results:")
        print("=" * 30)
        print(f"📝 Total Queries: {metrics['total_queries']}")
        print(f"🔍 Average Search Attempts: {metrics['avg_search_attempts']:.2f}")
        print(f"✅ Answers Found Rate: {metrics['answers_found_rate']:.2%}")
        print(f"\n🎯 Overall ROUGE Scores:")
        print(f"   ROUGE-1: {metrics['rouge_scores']['rouge1']:.4f}")
        print(f"   ROUGE-2: {metrics['rouge_scores']['rouge2']:.4f}")
        print(f"   ROUGE-L: {metrics['rouge_scores']['rougeL']:.4f}")
        
        # Show category-based results
        if metrics['category_metrics']:
            print(f"\n📋 Category-based Results:")
            print("=" * 30)
            for category, cat_metrics in metrics['category_metrics'].items():
                print(f"\n🏷️  {category}:")
                print(f"   📝 Queries: {cat_metrics['total_queries']}")
                print(f"   🔍 Avg Search Attempts: {cat_metrics['avg_search_attempts']:.2f}")
                print(f"   ✅ Answers Found Rate: {cat_metrics['answers_found_rate']:.2%}")
                print(f"   🎯 ROUGE-L: {cat_metrics['rouge_scores']['rougeL']:.4f}")
        
        print("\n🎉 Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!") 