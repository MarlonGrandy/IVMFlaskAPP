import openai as ai
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
ai.api_key = "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"


def generate_gpt3_response(text):
    """
    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    completions = ai.Completion.create(
        engine='text-davinci-003',  # Determines the quality, speed, and cost.
        temperature=0.5,            # Level of creativity in the response
        prompt=text,                # prompt
        max_tokens=1000,             # Maximum tokens in the prompt AND response
        n=1,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )

    # Return the first choice's text
    return completions.choices[0].text


def cos_sim(text1, text2):
    # Create a TfidfVectorizer object to calculate the TF-IDF scores
    vec_object = TfidfVectorizer()

    # Calculate the TF-IDF scores for the two texts
    tfidf = vec_object.fit_transform([text1, text2])

    # Calculate the cosine similarity between the two texts
    cosine_sim = cosine_similarity(tfidf[0], tfidf[1])

    # Print the cosine similarity score
    return cosine_sim[0][0]


def semantic_sim(text1, text2):
    # defined transfromer to use
    model = SentenceTransformer('stsb-roberta-large')
    # gets text embeddings for both input texts
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    # calculates cosine similarity between the embeddings

    cosine_sim = util.pytorch_cos_sim(embedding1, embedding2)[0]

    # returns the cosine similarty
    return float(cosine_sim[0])


def main():
    # Read in paragraph data to generate prompts from
    prompts = pd.read_csv("data/SmithsonHolt_final.csv")
    compare_text = "The Island Project is a visionary initiative launched by the Holt/Smithson Foundation to invite artists to embark on a long-term journey of slow growth and careful attention, in order to create works of art that align with the values of the renowned Land-artists, Nancy Holt and Robert Smithson. This project centers around Little Fort Island, a small islet near the coast of Maine, which was acquired by Holt and Smithson in 1972, and which now symbolizes non-human personhood and reminds us of our interconnectedness with the natural world. The Island Project's first step is to acknowledge the island's inherent value and personhood by recognizing its right not to be owned and assigning it its own digital soul. The project then seeks to \"sense\" the island through a variety of analog and digital sensors, gathering real-time data from its vegetation, tides, wind, and water quality, among others. This data will inform the creation of works of art that celebrate decentralized cognitions and amplify care rather than coercion. The resulting works of art will manifest two types of behavior: relational and expressive, both complementary to each other, and both celebrating the unique inner particularities of non-human cognitions. Through the growth of an Interspecies Virtual Machine from this island, the Island Project provides an artistic response to the question of what Land Art can be in our times. It serves as a visionary example of how we can engage with the vast otherness that inhabits Earth and a profound reminder of our shared responsibility to care for our planet and all its inhabitants. The Island Project's values reflect those of Nancy Holt and Robert Smithson, emphasizing the importance of working with the natural world, rather than against it, and celebrating the beauty and wonder of all living things."
    # Initialize counter and empty list for prompt completion pairs
    counter = 0
    prompt_completion = []
    semantic_sims = []

    # Loop through each Smithson and Holt combination in the data
    for h in prompts['Holt']:
        for s in prompts['Smithson']:

            # Create a prompt string combining the two texts
            prompt = f'Write one artistic paragraph interweaving the styles, themes, and diction of this text: "{h}" and this text: "{s}". Use an equal amount of details from both texts.'

            # Generate a response to the prompt using gpt-3 API
            completion = generate_gpt3_response(prompt)
            print(completion)

            # If cos similarity is between 0.6 and 0.8 for both originial texts and the completion, use the completion
            if cos_sim(s, completion) >= 0.4 and cos_sim(s, completion) <= 0.8 and cos_sim(h, completion) >= 0.4 and cos_sim(h, completion) <= 0.8:

                # append semntic similarity
                semantic_sims.append(
                    semantic_sim(compare_text, completion))

                # Append the prompt and completion pair to a list
                prompt_completion.append([prompt, completion])

            # Increment the counter and print progress
            counter += 1
            print(f"{counter} out of {len(prompts)*len(prompts)}")
            print(f"usable prompts: {len(prompt_completion)}/{counter}")

    combined = zip(semantic_sims, prompt_completion)  # combine the two lists
    sorted_combined = sorted(combined)  # sort the combined list based on b
    top_k = [elem[1] for elem in sorted_combined]  # extract the sorted a

    # Convert the prompt completion list into a pandas DataFrame and save it as a CSV file
    df = pd.DataFrame(top_k, columns=['prompt', 'completion'])
    df.to_csv(os.path.join("data", "completions.csv"), index=False)
    df_sims = pd.DataFrame(semantic_sims, columns=['semantic_sim'])
    df_sims.to_csv(os.path.join("data", "semantic_sims.csv"), index=False)


if __name__ == "__main__":
    main()
