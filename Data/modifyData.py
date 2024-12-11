from sqlalchemy import create_engine, MetaData, text

# Specify the database path
db_path = r'D:\AI\Chemical application\BioApp.db'

# Connect to the database
engine = create_engine(f'sqlite:///{db_path}')
connection = engine.connect()
metadata = MetaData()

# Reflect the existing database
metadata.reflect(bind=engine)

# Add the new column 'payment' to the User table
connection.execute(text('ALTER TABLE User ADD COLUMN payment FLOAT DEFAULT 0'))

# Ensure all existing users have the payment set to 0
connection.execute(text('UPDATE User SET payment = 0'))

# Verify the changes
result = connection.execute(text("PRAGMA table_info(User)"))
print("Updated schema for User table:")
for row in result:
    print(row)

# Verify data
data = connection.execute(text('SELECT ID, FullName, payment FROM User'))
print("\nUpdated data in User table:")
for row in data:
    print(row)

# Close the connection
connection.close()
