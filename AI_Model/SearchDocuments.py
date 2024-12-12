import torch
import spacy
import wikipedia
import requests
from xml.etree import ElementTree
import nltk
from nltk.tokenize import sent_tokenize
import wikipedia
import requests
from xml.etree import ElementTree
import torch
from transformers import AutoTokenizer
import wikipedia
import requests
from xml.etree import ElementTree
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os

class MiniSentenceTransformer:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L12-v2'):
        """
        Initialize the MiniSentenceTransformer with a pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model from HuggingFace Hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Perform mean pooling on the token embeddings, considering the attention mask.

        Args:
            model_output: Output from the transformer model.
            attention_mask: Attention mask for the input tokens.

        Returns:
            Tensor: Pooled sentence embeddings.
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        """
        Compute sentence embeddings for the given sentences.

        Args:
            sentences (list of str): Sentences to compute embeddings for.

        Returns:
            Tensor: Normalized sentence embeddings.
        """
        # Tokenize the sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
    

class RetrieveInformation:
    """
    A class to retrieve and process information from Wikipedia and arXiv.
    """

    def __init__(self, chunk_size=200):
        """
        Initialize the class with field-specific keywords and chunk size.

        Args:
            chunk_size (int): Approximate size of each chunk in words.
        """
        self.field_keywords = [
            "Artificial Intelligence", "Machine Learning", "Deep Learning", "Neural Networks",
            "Natural Language Processing", "Computer Vision", "Reinforcement Learning", "AI Ethics",
            "AI Algorithms", "AI Models", "Autonomous Systems", "AI-powered Automation",
            "Neural Architecture Search", "Generative Models", "AI in Pattern Recognition",
            "Bayesian Networks", "Speech Recognition", "Biotechnology", "Genetics", "CRISPR",
            "Gene Editing", "Artificial General Intelligence", "AI Consciousness", "Medical AI"
        ]
        
        # Initialize the BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.model = MiniSentenceTransformer()
        self.embedding_keywords = []
        for keyword in self.field_keywords:
            ebd = self.model.get_embeddings(keyword)
            self.embedding_keywords.append(ebd)
        self.embedding_keywords = torch.cat(self.embedding_keywords, dim=0)
        self.chunk_size = chunk_size

    def fetch_wikipedia_content(self, keyword):
        try:
            # Fetch Wikipedia page content
            page = wikipedia.page(keyword)
            content = page.content
            url = page.url

            # Tokenize content into sentences using BERT tokenizer
            # Tokenizer split by punctuation marks (e.g., period, question mark, etc.)
            sentences = self.tokenizer.tokenize(content)  # Tokenizes into word-level tokens
            sentences = self.tokenizer.convert_tokens_to_string(sentences).split('.')  # Simple sentence split

            # Filter sentences based on field-specific keywords
            filtered_sentences = [
                sent for sent in sentences
                if any(field_keyword.lower() in sent.lower() for field_keyword in self.field_keywords)
            ]

            # Split content into chunks of specified size
            chunks = []
            current_chunk = ""
            for sentence in filtered_sentences:
                if len(current_chunk.split()) + len(sentence.split()) <= self.chunk_size:
                    current_chunk += " " + sentence
                else:
                    chunks.append((current_chunk.strip(), url))
                    current_chunk = sentence
            if current_chunk:  # Add the remaining chunk
                chunks.append((current_chunk.strip(), url))

            return chunks, []  # Return content chunks and no disambiguation options

        except wikipedia.exceptions.DisambiguationError as e:
            return [], e.options  # Return disambiguation options if ambiguous
        except wikipedia.exceptions.PageError:
            return [], [f"No Wikipedia page found for '{keyword}'."]

    def fetch_arxiv_papers(self, query, max_results=5):
        url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',  # Search across all categories
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        response = requests.get(url, params=params)
        papers = []
        if response.status_code == 200:
            # Parse the XML response
            tree = ElementTree.fromstring(response.content)
            for entry in tree.findall("{http://www.w3.org/2005/Atom}entry"):
                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                authors = [author.text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
                published = entry.find("{http://www.w3.org/2005/Atom}published").text
                arxiv_url = entry.find("{http://www.w3.org/2005/Atom}id").text

                papers.append({
                    "title": title,
                    "published": published,
                    "summary": summary.strip(),
                    "url": arxiv_url
                })
        else:
            print(f"Failed to fetch arXiv data: {response.status_code}")
        return papers

    def find_keywords(self, sentences, k=3):
        embedding = self.model.get_embeddings(sentences)
        similarity = torch.matmul(embedding, self.embedding_keywords.T)
        top_k = torch.topk(similarity, k=k)
        indices = top_k.indices.tolist()
        key = [self.field_keywords[i] for i in indices[0]]
        return key

    def search_document(self, query):
        key = self.find_keywords(query)
        results = {'wikipedia': [], 'arxiv': []}
        for keyword in key:
            chunks, disambiguation_options = self.fetch_wikipedia_content(keyword)
            results['wikipedia'].append({
                'keyword': keyword,
                'chunks': chunks,
                'disambiguation': disambiguation_options
            })
            # Fetch arXiv papers
            arxiv_results = self.fetch_arxiv_papers(keyword)
            results['arxiv'].append({
                'keyword': keyword,
                'papers': arxiv_results
            })
        return self.post_process_results(results)
    def clean_text(self,text):
     text = " ".join(text.split())
     if text and text[0].islower():
        text = text[0].upper() + text[1:]
     return text
    def post_process_results(self,results):
     for wiki_entry in results.get('wikipedia', []):
         wiki_entry['keyword'] = self.clean_text(wiki_entry['keyword'])
         wiki_entry['chunks'] = [(self.clean_text(chunk), url) for chunk, url in wiki_entry['chunks']]
         wiki_entry['disambiguation'] = [self.clean_text(option) for option in wiki_entry['disambiguation']]
     for arxiv_entry in results.get('arxiv', []):
        for paper in arxiv_entry.get('papers', []):
            paper['title'] = self.clean_text(paper['title'])
            paper['summary'] = self.clean_text(paper['summary'])
            paper['url'] = paper['url'].strip()  # Ensure URLs are clean
     return results
            










#retriever=RetrieveInformation()

#query='I want to explore the development of AI in biology'
#results=retriever.search_document(query)
#print(results['arxiv'][0]['papers'])