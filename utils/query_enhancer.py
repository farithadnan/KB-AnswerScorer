import re

from utils.evaluation_utils import assess_response_quality

class QueryEnchancer:
    def __init__(self):
        """
        Initialize the QueryEnchancer with simplified rule-based enhancement.
        No ML models are needed for basic query enhancement.
        """
        self.common_issues = {
            # Basic issues
            "can't open": "I can't open the WhatsApp API application. When I try to launch it, nothing happens. Please provide step-by-step troubleshooting instructions, including any uninstall/reinstall steps if needed.",
            "won't open": "The WhatsApp API won't open on my system. I need detailed troubleshooting steps to fix this issue.",
            "not opening": "The WhatsApp API application is not opening. I need detailed steps to resolve this issue.",
            
            # Error messages and issues from documentation
            "error": "I'm getting an error message when using WhatsApp API. Need specific troubleshooting steps.",
            "not working": "WhatsApp API is not working properly. What are the step-by-step instructions to resolve this?",
            "port 8000": "I'm getting a 'port 8000 already in use' error with WhatsApp API. How do I fix this?",
            "session folder": "Having issues with the WhatsApp API session folder. Need steps to resolve this.",
            "ENVIRONMENT": "Getting an ENVIRONMENT error with WhatsApp API. How do I fix this?",
            
            # Specific functionality issues
            "qr code": "Having issues with the QR code in WhatsApp API. It's not displaying properly or can't be scanned.",
            "scan": "Can't scan the QR code in WhatsApp API. Need detailed troubleshooting steps.",
            "install": "Need detailed steps to properly install WhatsApp API on my system.",
            "uninstall": "Need step-by-step instructions to completely uninstall WhatsApp API and reinstall it cleanly.",
            "delete": "How do I properly delete WhatsApp API session data?",
            "restart": "Do I need to restart my system or VM after installing WhatsApp API?",
            "vm": "Having issues with WhatsApp API on my Virtual Machine. What are the troubleshooting steps?",
            
            # More specific errors from documentation
            "crash": "WhatsApp API keeps crashing when I try to use it. Need step-by-step solution.",
            "port": "Getting port conflict issues with WhatsApp API. How do I resolve this?",
            "api key": "Issues with API key configuration in WhatsApp API. What are the steps to fix this?",
            "fileserver": "Can't access the fileserver.mzm location for WhatsApp API installation.",
            "add or remove": "Need to use Add or Remove Programs for WhatsApp API? What are the steps?",
            "schedule-send": "Having issues with schedule-send feature in WhatsApp API."
        }

    def pre_process(self, query):
        """
        Pre-process the user query to enhance it for better understanding and response.
        This includes adding context, rephrasing, and ensuring clarity. 
        
        Args:
            query (str): The original user query.
            
        Returns:
            str: The enhanced query.
        """
        query_lower = query.lower()
        
        # Direct match with common issue patterns
        for pattern, enhanced in self.common_issues.items():
            if pattern in query_lower:
                return enhanced
        
        # For very short queries without context
        if len(query.split()) < 3:
            return f"I'm having this issue with WhatsApp API: {query}. Please provide detailed troubleshooting steps including any uninstall, reinstall, or system restart instructions that might be needed."
        
        # For queries that lack context, add some domain-specific context
        if "whatsapp" not in query_lower and "api" not in query_lower:
            return f"In the context of WhatsApp API technical support, I'm having this issue: {query}. Please provide step-by-step troubleshooting instructions."
        
        # For queries with domain context, but lacking specificity, add structure
        if not re.search(r'(how|why|what|when|where|which|who|is|are|can|could|should)', query_lower):
            return f"How do I resolve this WhatsApp API issue: {query}? Please provide detailed step-by-step instructions including any uninstall/reinstall steps if needed."
        
        # If the query already seems well-formed, leave it as is
        return query
    
    def post_process(original_prompt, metrics, 
                        bert_threshold=0.5, 
                        f1_threshold=0.3, 
                        bleu_threshold=0.1,
                        combined_threshold=0.4):
        """
        Post-process the original prompt based on evaluation metrics.
        
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