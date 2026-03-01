import pandas as pd

df = pd.read_csv('DATASETS/KAGGLE MILK DATA/dairy_dataset.csv')

# Milk only
milk = df[df['Product Name']=='Milk']
print(f"=== MILK ONLY ({len(milk)} rows) ===")
print(f"Qty stats:\n{milk['Quantity (liters/kg)'].describe()}")
print(f"\nPrice stats:\n{milk['Price per Unit'].describe()}")
print(f"\nShelf life:\n{milk['Shelf Life (days)'].value_counts()}")
print(f"\nStorage:\n{milk['Storage Condition'].value_counts()}")

# Brands
print(f"\n=== BRAND STATS ===")
for brand in ['Amul', 'Mother Dairy', 'Sudha']:
    b = df[df['Brand']==brand]
    print(f"\n{brand}: {len(b)} rows")
    print(f"  Products: {b['Product Name'].unique()}")
    print(f"  Locations: {b['Location'].unique()}")
    print(f"  Avg Qty: {b['Quantity (liters/kg)'].mean():.1f}")
    print(f"  Avg Price: {b['Price per Unit'].mean():.2f}")
    print(f"  Avg Stock: {b['Quantity in Stock (liters/kg)'].mean():.1f}")
    print(f"  Sales Channels: {b['Sales Channel'].value_counts().to_dict()}")

# All products stats
print(f"\n=== ALL PRODUCTS ===")
for prod in df['Product Name'].unique():
    p = df[df['Product Name']==prod]
    print(f"{prod}: {len(p)} rows, avg_qty={p['Quantity (liters/kg)'].mean():.1f}, avg_price={p['Price per Unit'].mean():.2f}, shelf={p['Shelf Life (days)'].mean():.0f}")
