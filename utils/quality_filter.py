def assess_response_quality(metrics, 
                           bert_threshold=0.5, 
                           f1_threshold=0.3, 
                           bleu_threshold=0.1):
    """
    Assess the quality of a response based on evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        bert_threshold: Minimum acceptable BERTScore
        f1_threshold: Minimum acceptable traditional F1 score
        bleu_threshold: Minimum acceptable BLEU score
        
    Returns:
        tuple: (is_acceptable, feedback_message)
    """
    bert_score = metrics['bert_f1']
    f1_score = metrics['trad_f1']
    bleu_score = metrics['bleu']

    is_acceptable = True
    feedback = []

    if bert_score < bert_threshold:
        is_acceptable = False
        feedback.append(f"BERTScore ({bert_score:.4f}) below threshold ({bert_threshold:.2f})")
    
    if f1_score < f1_threshold:
        is_acceptable = False
        feedback.append(f"F1 score ({f1_score:.4f}) below threshold ({f1_threshold:.2f})")
    
    if bleu_score < bleu_threshold:
        is_acceptable = False
        feedback.append(f"BLEU score ({bleu_score:.4f}) below threshold ({bleu_threshold:.2f})")
    
    if is_acceptable:
        return True, "Response meets quality standards."
    else:
        feedback_message = "Response quality issues detected:\n- " + "\n- ".join(feedback)
        feedback_message += "\n\nSuggested actions:\n"
        
        if bert_score < bert_threshold:
            feedback_message += "- Improve semantic relevance to the question\n"
        if f1_score < f1_threshold:
            feedback_message += "- Include more key terms from the reference solution\n"
        if bleu_score < bleu_threshold:
            feedback_message += "- Structure the response more similarly to reference solutions\n"
            
        feedback_message += "\nPlease provide more detailed context or a more specific query."
        
        return False, feedback_message
    

def get_improved_prompt(original_prompt, metrics, 
                        bert_threshold=0.5, 
                        f1_threshold=0.3, 
                        bleu_threshold=0.1):
    """
    Generate an improved prompt if the response quality is below thresholds.
    
    Args:
        original_prompt: The original prompt text
        metrics: Evaluation metrics dictionary
        thresholds: As in assess_response_quality
        
    Returns:
        str: Either the original prompt (if quality is good) or an enhanced prompt
    """
    is_acceptable, feedback = assess_response_quality(
        metrics, bert_threshold, f1_threshold, bleu_threshold
    )
    
    if is_acceptable:
        return original_prompt
    
     # Create an enhanced prompt
    enhanced_prompt = (
        f"I need more detailed information about this issue. "
        f"My previous question was:\n\n{original_prompt}\n\n"
        f"Please provide a comprehensive answer with:\n"
        f"- Step-by-step instructions\n"
        f"- Relevant technical details\n"
        f"- Any specific commands or settings needed\n"
        f"- Common pitfalls to avoid"
    )
    return enhanced_prompt
