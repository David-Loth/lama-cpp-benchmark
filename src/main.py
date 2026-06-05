import sys
from pathlib import Path
from basic.bench_chat import run_chat_completion_with_metrics
from basic.llama_cpp_api import initialize_llm


def main():
    if len(sys.argv)==1:
        path_to_models=Path('.')/Path("models")
    else:
        path_to_models=sys.argv[1]
    model_file_paths=list(path_to_models.glob('**/*.gguf'))
    print(model_file_paths)

    test_chat_messages = [
    [{"role": "user", "content": "Bonjour ! Est-ce que tu fonctionnes correctement ?"}],
    [{"role": "user", "content": "Explique-moi la différence entre le HTML et le CSS en deux phrases."}],
    [{"role": "user", "content": "Résous cette équation : 3x + 5 = 20. Donne-moi la valeur de x."}]
]

    metrics={str(model):[] for model in model_file_paths}
    avgmetric=dict()
    print()

    for model in model_file_paths:
        llm= initialize_llm(str(model))

        for message in test_chat_messages:
            metric=run_chat_completion_with_metrics(llm,message)
            print(metric)
            metrics[str(model)].append(metric)

        avgmetric[str(model)]=dict()
        tokens=0
        gen_time=0
        ttft_ms=0
        for m in metrics[str(model)]:
            tokens+=m["total_tokens"]
            gen_time+=m["gen_time"]
            ttft_ms+=m["ttft_ms"]

        avgmetric[str(model)]["t/s"]=tokens/gen_time
        avgmetric[str(model)]["avg_ttft_ms"]=ttft_ms/len(test_chat_messages)
    
    print(avgmetric)



if __name__=="__main__":
    main()