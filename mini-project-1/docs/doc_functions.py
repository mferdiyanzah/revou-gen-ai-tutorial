from pdfminer.high_level import extract_text
import streamlit as st
import re

def doc_extractor():
    """Extract text from the PDF file"""
    file_path = 'docs/FAQ Dexa Medica.pdf'
    extracted_text = extract_text(file_path)
    return extracted_text

def doc_mapper(extracted_text):
    """Map document content by headers and parse Q&A pairs"""
    headers = ['Pertanyaan Umum Tentang Dexa Medica', 'Produk & Inovasi', 'Fasilitas & Produksi', 'Rekrutmen & Karier', 'Hubungan dan Komunikasi', 'Lain-lain / Tambahan', 'FAQ Tambahan Dexa Medica (Pertanyaan 21–50)']
    doc_map = {}
    
    # Split text into lines for easier processing
    lines = extracted_text.split('\n')
    current_header = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if current line is a header
        is_header = False
        for header in headers:
            if header in line:
                # If we were already processing a header, save its content
                if current_header and current_content:
                    doc_map[current_header] = parse_qa_pairs('\n'.join(current_content))
                
                # Start processing new header
                current_header = header
                current_content = []
                is_header = True
                break
        
        # If not a header and we have a current header, add to content
        if not is_header and current_header:
            current_content.append(line)
    
    # Don't forget to save the last header's content
    if current_header and current_content:
        doc_map[current_header] = parse_qa_pairs('\n'.join(current_content))
    
    if not doc_map:
        st.error("No headers found in the document")
    
    return doc_map

def parse_qa_pairs(content):
    """Parse content into question-answer pairs based on numbering"""
    qa_pairs = {}
    
    # First try the regex approach which works for most cases
    question_pattern = r'(\d+)\.\s+([^\n?]+\?)([\s\S]*?)(?=\n\d+\.\s+[^\n?]+\?|$)'
    matches = re.findall(question_pattern, content, re.DOTALL)
    
    if matches:
        for match in matches:
            question_num = int(match[0])
            question_text = match[1].strip()
            answer_text = match[2].strip()

            # Clean up the question text
            if not question_text.endswith('?'):
                question_text += '?'
            
            qa_pairs[question_text] = answer_text
    
    # For any questions with empty answers, try to find them elsewhere in the content
    empty_answer_questions = [q for q, a in qa_pairs.items() if not a.strip()]
    
    if empty_answer_questions:
        lines = content.split('\n')
        
        for question in empty_answer_questions:
            # Look for answer patterns in the entire content
            answer_found = False
            
            # Extract key terms from the question to search for answers
            if 'arti nama' in question.lower() and 'dexa' in question.lower():
                # Look for the specific answer pattern
                for line in lines:
                    line_clean = line.strip()
                    if ('nama' in line_clean.lower() and 'dexa' in line_clean.lower() and 
                        'berasal' in line_clean.lower() and 'yunani' in line_clean.lower()):
                        answer_parts = [line_clean]
                        
                        # Look for the continuation on the next line
                        line_idx = lines.index(line)
                        if line_idx + 1 < len(lines):
                            next_line = lines[line_idx + 1].strip()
                            if next_line and 'kesempurnaan' in next_line.lower():
                                answer_parts.append(next_line)
                        
                        qa_pairs[question] = ' '.join(answer_parts)
                        answer_found = True
                        break
            
            # If still no answer found, keep the empty answer
            if not answer_found:
                qa_pairs[question] = ""
    
    # If no numbered questions found, return the content as is
    if not qa_pairs:
        return content
    
    return qa_pairs

def create_question_chunks(doc_map):
    """Create chunks based on individual questions"""
    chunks = []
    chunk_id = 1
    
    for category, qa_pairs in doc_map.items():
        if isinstance(qa_pairs, dict):
            for question, answer in qa_pairs.items():
                chunk = {
                    'chunk_id': chunk_id,
                    'category': category,
                    'question': question,
                    'answer': answer,
                    'content': f"Kategori: {category}\n\nPertanyaan: {question}\n\nJawaban: {answer}",
                    'metadata': {
                        'source': 'FAQ Dexa Medica.pdf',
                        'category': category,
                        'type': 'question_answer'
                    }
                }
                chunks.append(chunk)
                chunk_id += 1
        else:
            # Handle cases where qa_pairs is not a dictionary (fallback)
            chunk = {
                'chunk_id': chunk_id,
                'category': category,
                'question': None,
                'answer': None,
                'content': f"Kategori: {category}\n\n{qa_pairs}",
                'metadata': {
                    'source': 'FAQ Dexa Medica.pdf',
                    'category': category,
                    'type': 'general_content'
                }
            }
            chunks.append(chunk)
            chunk_id += 1
    
    return chunks

def get_chunk_summaries(chunks):
    """Get summary information about the chunks"""
    summary = {
        'total_chunks': len(chunks),
        'categories': {},
        'questions_per_category': {}
    }
    
    for chunk in chunks:
        category = chunk['category']
        if category not in summary['categories']:
            summary['categories'][category] = 0
            summary['questions_per_category'][category] = []
        
        summary['categories'][category] += 1
        if chunk['question']:
            summary['questions_per_category'][category].append(chunk['question'])
    
    return summary 