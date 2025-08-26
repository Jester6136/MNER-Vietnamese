import requests
from urllib.parse import quote
import re
from transformers import AutoTokenizer
import time

class CoNLL2003WithContextProcessor:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512
    
    def read_conll_data(self, file_path):
        """Read CoNLL-2003 format data (Token NER)"""
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":  # Empty line indicates end of sentence
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token, ner_tag = parts[0], parts[1]
                        current_sentence.append((token, ner_tag))
            
            # Don't forget the last sentence if file doesn't end with empty line
            if current_sentence:
                sentences.append(current_sentence)
        
        return sentences
    
    def search_google(self, sentence_tokens, num_results=10):
        """Get top 10 relevant information from Google search"""
        # Join tokens to form the search query
        query = " ".join([token for token, _ in sentence_tokens])
        
        # For demonstration - in practice you'd use Google Custom Search API
        # This is a simplified version - you should replace with actual API call
        try:
            # Example using Google Custom Search API (you need API key)
            # search_url = f"https://www.googleapis.com/customsearch/v1"
            # params = {
            #     'key': 'YOUR_API_KEY',
            #     'cx': 'YOUR_SEARCH_ENGINE_ID',
            #     'q': query,
            #     'num': num_results
            # }
            # response = requests.get(search_url, params=params)
            # results = response.json()
            
            # Mock response for demonstration
            external_context = f"External context about: {query}. This is mock data from Google search results. Additional information relevant to the query."
            return external_context
            
        except Exception as e:
            print(f"Search error: {e}")
            return "External context placeholder due to search error."
    
    def truncate_to_max_length(self, sentence_text, external_context, max_length=512):
        """Truncate combined text to max_length tokens"""
        # Tokenize sentence and external context separately to understand their token counts
        sentence_tokens = self.tokenizer.tokenize(sentence_text)
        context_tokens = self.tokenizer.tokenize(external_context)
        
        total_tokens = len(sentence_tokens) + 1 + len(context_tokens)  # +1 for <EOS>
        
        if total_tokens <= max_length:
            return sentence_text, external_context
        else:
            # Calculate how many tokens we can keep
            available_tokens = max_length - len(sentence_tokens) - 1  # -1 for <EOS>
            if available_tokens > 0:
                # Truncate external context
                truncated_context_tokens = context_tokens[:available_tokens]
                truncated_context = self.tokenizer.convert_tokens_to_string(truncated_context_tokens)
                return sentence_text, truncated_context
            else:
                # Very long sentence, truncate both
                sentence_tokens_truncated = sentence_tokens[:max_length-10]  # Reserve some space
                sentence_text_truncated = self.tokenizer.convert_tokens_to_string(sentence_tokens_truncated)
                context_tokens_truncated = context_tokens[:9]  # Very limited context
                context_text_truncated = self.tokenizer.convert_tokens_to_string(context_tokens_truncated)
                return sentence_text_truncated, context_text_truncated
    
    def process_sentence(self, sentence_tokens):
        """Process a single sentence through the pipeline"""
        # Convert tokens to text for search
        sentence_text = " ".join([token for token, _ in sentence_tokens])
        
        # Get external context from Google search
        external_context = self.search_google(sentence_tokens)
        
        # Truncate to fit within token limit
        sentence_text, external_context = self.truncate_to_max_length(
            sentence_text, external_context, self.max_length
        )
        
        return sentence_text, external_context
    
    def create_enhanced_dataset(self, sentences):
        """Create the enhanced dataset with external context"""
        enhanced_data = []
        
        for sentence in sentences:
            # Process the sentence
            sentence_text, external_context = self.process_sentence(sentence)
            
            # Create the enhanced format
            enhanced_sentence = []
            
            # Add original sentence tokens with their NER tags
            for token, ner_tag in sentence:
                enhanced_sentence.append((token, ner_tag))
            
            # Add EOS token
            enhanced_sentence.append(("<EOS>", "E"))
            
            # Add external context tokens with E tag
            context_tokens = self.tokenizer.tokenize(external_context)
            for token in context_tokens:
                enhanced_sentence.append((token, "E"))
            
            enhanced_data.append(enhanced_sentence)
        
        return enhanced_data
    
    def save_enhanced_dataset(self, enhanced_data, output_file):
        """Save the enhanced dataset in CoNLL format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in enhanced_data:
                for token, tag in sentence:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")  # Empty line between sentences

# Usage example
def main():
    processor = CoNLL2003WithContextProcessor()
    
    # Read your CoNLL data
    sentences = processor.read_conll_data("your_dataset.conll")
    
    # Process all sentences
    enhanced_data = processor.create_enhanced_dataset(sentences)
    
    # Save the result
    processor.save_enhanced_dataset(enhanced_data, "enhanced_dataset.conll")
    
    # Print first sentence as example
    if enhanced_data:
        print("First enhanced sentence:")
        for token, tag in enhanced_data[0]:
            print(f"{token}\t{tag}")

if __name__ == "__main__":
    main()