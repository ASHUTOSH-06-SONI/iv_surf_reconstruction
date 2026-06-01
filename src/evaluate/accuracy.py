import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load submission
submission = pd.read_csv("/Users/santoshsoni/Desktop/NKSR_Hackathon/submission.csv")

print("Basic statistics:")
print(f"Total predictions: {len(submission)}")
print(f"Unique underlyings: {submission['underlying'].nunique()}")
print(f"Call options: {len(submission[submission['option_type'] == 0])}")
print(f"Put options: {len(submission[submission['option_type'] == 1])}")

print(f"\nIV prediction statistics:")
print(f"Min predicted IV: {submission['predicted_iv'].min():.4f}")
print(f"Max predicted IV: {submission['predicted_iv'].max():.4f}")
print(f"Mean predicted IV: {submission['predicted_iv'].mean():.4f}")
print(f"Median predicted IV: {submission['predicted_iv'].median():.4f}")
print(f"Std predicted IV: {submission['predicted_iv'].std():.4f}")

# Check for reasonable IV values (typically 0.1 to 3.0 for most options)
unrealistic_low = len(submission[submission['predicted_iv'] < 0.05])
unrealistic_high = len(submission[submission['predicted_iv'] > 5.0])
print(f"\nPotential Issues:")
print(f"Very low IV (<5%): {unrealistic_low} predictions")
print(f"Very high IV (>500%): {unrealistic_high} predictions")

# Distribution plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(submission['predicted_iv'], bins=50, alpha=0.7)
plt.xlabel('Predicted IV')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted IV')
plt.grid(True)

plt.subplot(2, 2, 2)
calls = submission[submission['option_type'] == 0]
puts = submission[submission['option_type'] == 1]
plt.hist(calls['predicted_iv'], bins=30, alpha=0.7, label='Calls')
plt.hist(puts['predicted_iv'], bins=30, alpha=0.7, label='Puts')
plt.xlabel('Predicted IV')
plt.ylabel('Frequency')
plt.title('IV by Option Type')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
# IV vs Strike Price pattern
plt.scatter(submission['strike_price'], submission['predicted_iv'], alpha=0.5)
plt.xlabel('Strike Price')
plt.ylabel('Predicted IV')
plt.title('IV vs Strike Price')
plt.grid(True)

plt.subplot(2, 2, 4)
# Box plot by underlying (if not too many)
if submission['underlying'].nunique() <= 10:
    submission.boxplot(column='predicted_iv', by='underlying', ax=plt.gca())
    plt.title('IV Distribution by Underlying')
    plt.xlabel('Underlying')
    plt.ylabel('Predicted IV')
else:
    # Just show top 5 underlyings by count
    top_underlyings = submission['underlying'].value_counts().head(5).index
    subset = submission[submission['underlying'].isin(top_underlyings)]
    subset.boxplot(column='predicted_iv', by='underlying', ax=plt.gca())
    plt.title('IV Distribution (Top 5 Underlyings)')

plt.tight_layout()
plt.show()

print(f"\nSanity check completed")
print(f"Typical IV ranges: 10%-100% for most stocks, 15%-80% for indices")
