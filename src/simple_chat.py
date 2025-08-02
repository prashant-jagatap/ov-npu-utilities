import openvino_genai as ov_genai

def RunSimpleChat():
    model_dir = "./OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov"
    pipe = ov_genai.LLMPipeline(model_dir, device="NPU")  # pyright: ignore[reportArgumentType]
    
    while True:
        pipe.start_chat()

        user_input = input("User: ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        
        config = pipe.get_generation_config()
        config.max_new_tokens = 256
        
        answer = pipe.generate(user_input, config = config)
        print(f"Assistant: {answer}\n")
        
        pipe.finish_chat()

RunSimpleChat()