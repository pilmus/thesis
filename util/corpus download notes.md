- For 2019 track, went back to 16 Aug 2019 snapshot of [https://api.semanticscholar.org/corpus/download](https://web.archive.org/web/20190816155036/https://api.semanticscholar.org/corpus/download/) and downloaded corpus with 

	```sh
	wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2019-01-31/manifest.txt
	wget -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2019-01-31/ -i manifest.txt
	```

	so I would have the corpus as it was available at the time the track was occurring (participant instructions were published 13 Aug 2019).
	
- Wrote shell script to retrieve all released manifests for 2020, but they all have more than 47 files and the participant instructions say you only have 47 files, so we'll assume 2020 uses the same corpus as 2019.