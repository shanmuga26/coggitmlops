import os 
import pandas as pd
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__),'..', '.env')
load_dotenv(env_path)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def fraud_detection_train(file_path):
    df=pd.read_csv(file_path)
    df.dropna(inplace=True)
    x=df.drop([['transaction_id','label']],axis=1)
    y=df['label']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    clf=DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=(y_pred==y_test).mean()
    print(f"Model Accuracy: {accuracy:.2f}")
    f1_score_value=f1_score(y_test,y_pred)
    print(f"Model F1 Score: {f1_score_value:.2f}")
    precision_value=precision_score(y_test,y_pred)
    recall_value=recall_score(y_test,y_pred)
    print(f"Model Precision: {precision_value:.2f}")
    print(f"Model Recall: {recall_value:.2f}")

    cm=confusion_matrix(y_test,y_pred)
    print("Confusion Matrix:")
    print(cm)
if __name__ == "__main__":
    file_path=os.getenv('file_path')
    fraud_detection_train(file_path)