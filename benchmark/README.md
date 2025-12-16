The benchmark to beat is polars-fastembed with `Xenova/all-MiniLM-L6-v2`,
in which embedding and retrieval of the MiniLM-L6-v2 on all PEPs took 30s on CPU,
20s on GPU.

> - Total token count: 3,615,903 = 3.6M tokens in 30s = 8ms per 1k tokens
>
>   - Roughly 2x as fast as recently reported 14.7ms per 1k tokens [here](https://www.reddit.com/r/LocalLLaMA/comments/1nrgklt/opensource_embedding_models_which_one_to_use/) or [here](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)
>
> - On GPU, this falls to 5.3ms per 1k tokens (20s total time to run), with all cores still used for a
>   significant portion of the computation

The same operation (copied over from the polars-fastembed repo) takes 1.8s,
so approx 0.5ms per 1k tokens!
