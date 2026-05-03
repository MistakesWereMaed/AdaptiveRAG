# Check server is running
curl http://localhost:9200

# Index DPR Wiki
python ../src/build_index/main.py wiki

# Index HotpotQA
python ../src/build_index/main.py hotpotqa

# Index musique
python ../src/build_index/main.py musique

# Index 2wikimultihopqa
python ../src/build_index/main.py 2wikimultihopqa

# Check outputs
curl "http://localhost:9200/wiki/_count?pretty"
curl "http://localhost:9200/hotpotqa/_count?pretty"
curl "http://localhost:9200/musique/_count?pretty"
curl "http://localhost:9200/2wikimultihopqa/_count?pretty"