from sentence_transformers import SentenceTransformer, util
import pandas as pd
from numpy import dot
from numpy.linalg import norm


# Calculate cosine similarity
def cos_sim(num1, num2):
    return dot(num1, num2) / (norm(num1) * norm(num2))


def return_top3(mentee_question, df_data_param):
    embedding = model.encode(mentee_question)
    df_data_param['score'] = df_data_param.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    # df_data['score'] = df_data.apply(lambda x: util.pytorch_cos_sim(x['embedding'], embedding), axis=1)

    # Top 3의 결과를 반환
    return df_data_param.sort_values(by='score', ascending=False).head(3)


if __name__ == "__main__":
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 멘티의 질문을 Spring Boot 로부터 받는다; 추후 구현
    sentence_from_mentee = "Here is a test sentence from Mentee!"

    # 데이터베이스에서 id, en_data, answer 등의 데이터를 불러와서 exampleData.csv 의 양식에 추가; 추후 구현
    df_data = pd.read_csv("exampleData.csv")
    # print(df_data.columns.tolist()) # Check column name
    df_data['embedding'] = df_data.apply(lambda row: model.encode(row.en_data), axis=1)

    print(return_top3(sentence_from_mentee, df_data))
