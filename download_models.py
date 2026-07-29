from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

AutoTokenizer.from_pretrained("facebook/m2m100_418M")
AutoModelForSeq2SeqLM.from_pretrained("Jayyydyyy/m2m100_418m_tokipona")
AutoTokenizer.from_pretrained("gpt2")
AutoModelForCausalLM.from_pretrained("adamthekiwi/toki-pona-gpt2-alpaca-best")
