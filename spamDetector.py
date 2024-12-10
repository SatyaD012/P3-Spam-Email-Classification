import streamlit as st
import pickle

model = pickle.load(open(r'D:\AI_internship\P3-Spam-Email-Classification-using-NLP-and-Machine-Learning-main\spam.pkl', 'rb'))
cv = pickle.load(open(r'D:\AI_internship\P3-Spam-Email-Classification-using-NLP-and-Machine-Learning-main\vectorizer.pkl', 'rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.subheader("Classification")
    user_input = st.text_area("Enter an email to classify", height=150)
    if st.button("Classify"):
        if user_input:
            data = [user_input]
            print(data)
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            st.write("Please enter an email to classify.")
            
main()
