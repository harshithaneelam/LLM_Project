import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Generate synthetic data
def generate_synthetic_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        data.append({
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address(),
            "company": fake.company(),
            "text": fake.paragraph()
        })
    
    return pd.DataFrame(data)

# Generate and save the dataset
df = generate_synthetic_data(1000)
df.to_csv("synthetic_dataset.csv", index=False)

print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")
