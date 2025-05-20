def assess_response_quality(metrics, 
                           bert_threshold=0.5, 
                           f1_threshold=0.3, 
                           bleu_threshold=0.1,
                           combined_threshold=0.4
    ):
    """
    Assess the quality of a response based on evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        bert_threshold: Minimum acceptable BERTScore
        f1_threshold: Minimum acceptable traditional F1 score
        bleu_threshold: Minimum acceptable BLEU score
        combined_threshold: Minimum acceptable combined score
        
    Returns:
        tuple: (is_acceptable, feedback_message)
    """
    is_acceptable = True
    feedback = []

    bert_score = metrics['bert_f1']
    f1_score = metrics['trad_f1']
    bleu_score = metrics['bleu']
    combined_score = metrics.get('combined_score', 0)


    if bert_score < bert_threshold:
        is_acceptable = False
        feedback.append(f"BERTScore ({bert_score:.4f}) below threshold ({bert_threshold:.2f})")
    
    if f1_score < f1_threshold:
        is_acceptable = False
        feedback.append(f"F1 score ({f1_score:.4f}) below threshold ({f1_threshold:.2f})")
    
    if bleu_score < bleu_threshold:
        is_acceptable = False
        feedback.append(f"BLEU score ({bleu_score:.4f}) below threshold ({bleu_threshold:.2f})")
    
    # Consider combined score as well
    if combined_score < combined_threshold:
        is_acceptable = False
        feedback.append(f"Combined score ({combined_score:.4f}) below threshold ({combined_threshold:.2f})")
    
    # But if combined score is very good, accept despite individual failures
    if combined_score >= combined_threshold * 1.25:
        is_acceptable = True
        feedback = [f"Combined score ({combined_score:.4f}) indicates good quality despite some low metrics"]
    
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
                        bleu_threshold=0.1,
                        combined_threshold=0.4):
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
        metrics, bert_threshold, f1_threshold, bleu_threshold, combined_threshold
    )
    
    if is_acceptable:
        return original_prompt
    
    # Extract specific issues from feedback
    bert_failed = metrics['bert_f1'] < bert_threshold
    f1_failed = metrics['trad_f1'] < f1_threshold
    bleu_failed = metrics['bleu'] < bleu_threshold
    combined_failed = metrics.get('combined_score', 0) < combined_threshold
    
    # Create an enhanced prompt
    enhanced_prompt = (
        f"I need more detailed information about this issue. "
        f"My previous question was:\n\n{original_prompt}\n\n"
        f"The previous response had these quality issues:\n{feedback}\n\n"
        f"Please provide a comprehensive answer with:\n"
    )
    
    # Add specific instructions based on which metrics failed
    if bert_failed:
        enhanced_prompt += "- Better semantic relevance to my specific question\n"
    if f1_failed:
        enhanced_prompt += "- More key technical terms and specific terminology\n"
    if bleu_failed:
        enhanced_prompt += "- Clearer structure similar to standard solutions\n"
    if combined_failed:
        enhanced_prompt += "- A more comprehensive and detailed response\n"
    
    # Add general improvements
    enhanced_prompt += (
        "- Step-by-step instructions\n"
        "- Relevant technical details\n"
        "- Any specific commands or settings needed\n"
        "- Common pitfalls to avoid"
    )
    return enhanced_prompt
