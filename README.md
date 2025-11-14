Build:
docker build  -f Dockerfile -t aessiari/iri-llm:m4-1 .  

Usage: 
python3 tiny_gpt2_cpu_1k.py <input> <outdir> <steps> <lines>

Example:
python3 tiny_gpt2_cpu_1k.py /IRI/synthetic_training_data.txt /tmp 1000 500
