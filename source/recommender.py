import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("recommender_model.keras")

# Load data
data_encoded = pd.read_csv('../data/preprocessed_music.csv')
data_original = pd.read_json('../data/Musical_Instruments_5.json', lines=True)

# Load encoded items data from dataframe
items_data = data_encoded['item_id_encoded']

# Load encoders
user_encoder = joblib.load('user_encoder.joblib')
item_encoder = joblib.load('item_encoder.joblib')

def get_recommendations(user_id, model, items_data, top_n=5, pool_size=20):
    try:
        # Ensure pool_size >= top_n
        pool_size = max(pool_size, top_n)
        
        # Create array with user_id repeated for all items
        users_array = np.array([user_id] * len(items_data))
        
        # Create array with all item ids
        items_array = items_data.to_numpy()
        
        # Predict the ratings for all user-item pairs
        predictions = model.predict([users_array, items_array])
        print(f'Predictions: {predictions}')
        
        # Get top pool_size item indices
        top_item_indices = predictions.flatten().argsort()[-pool_size:][::-1]
        print(f'Top item indices: {top_item_indices}')
        
        # Get corresponding item details
        recommended_items = items_data.iloc[top_item_indices]
        print(f'Recommended items: {recommended_items}')
        
        # Ensure the recommended items are unique
        unique_recommended_items = pd.Series(recommended_items).unique()[:top_n]
        print(f'Unique recommended items: {unique_recommended_items}')
        
        return unique_recommended_items
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")


# Streamlit UI
st.title('My Recommender System')

# Taking user ID as input
user_id_original = st.text_input("Enter your User ID:")

# When 'Get Recommendations' is pressed
if st.button('Get Recommendations'):
    # Validate user ID
    if user_id_original:
        try:
            # Encode user ID
            user_id = user_encoder.transform([user_id_original])[0]
            print('executed1')

            # Get recommendations
            recommended_items_encoded = get_recommendations(user_id, model, items_data, top_n=5)
            print('executed2')


            # Decode item IDs and fetch item details from original data
            print('Recommended items encoded:', recommended_items_encoded)
            recommended_items_ids_original = item_encoder.inverse_transform(recommended_items_encoded)
            print('Recommended items original ids:', recommended_items_ids_original)

            # Fetching and displaying item details
            recommended_items = data_original[data_original['asin'].isin(recommended_items_ids_original)]
            print('Recommended items:', recommended_items[['asin', 'reviewText']])

            # Display recommendations
            st.write("Top recommendations for you:")
            for item_id in recommended_items_ids_original:
                item_reviews = data_original[data_original['asin'] == item_id]

                # Ensure 'votes' is numeric and handle NaN values
                item_reviews['vote'] = pd.to_numeric(item_reviews['vote'], errors='coerce').fillna(0)

                # Selecting the most upvoted review for the item
                most_upvoted_review = item_reviews.loc[item_reviews['vote'].idxmax()]

                st.write(f"(Product ID: {most_upvoted_review['asin']})")
                st.write(f"{most_upvoted_review['reviewText']}")
                st.write('')
                st.write('')

            print('executed4')
            
        
        except Exception as e:
            st.write("An error occurred:", str(e))
            st.write("Please enter a valid User ID.")
    
    else:
        st.write("Please enter a User ID.")