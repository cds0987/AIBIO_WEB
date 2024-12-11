from sqlalchemy import create_engine, Column, String, Integer, ForeignKey
from sqlalchemy.types import Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'User'
    
    ID = Column(String(255), primary_key=True)  # Unique identifier for the user
    FullName = Column(String(255), nullable=False)  # User's full name
    Email = Column(String(255), nullable=False, unique=True)  # Unique email
    Password = Column(String(255), nullable=False)  # Password for authentication
    Level = Column(Integer, nullable=False)  # User's access level
    payment = Column(Float, default=0)
    tracking = relationship("Tracking", back_populates="user")

class Admin(Base):
    __tablename__ = 'Admin'
    
    ID = Column(String(255), primary_key=True)  # Unique identifier for admin
    FullName = Column(String(255), nullable=False)  # Admin's full name
    Email = Column(String(255), nullable=False, unique=True)  # Unique email for admin
    Password = Column(String(255), nullable=False)  # Password for admin authentication

class Model(Base):
    __tablename__ = 'Model'
    
    ModelName = Column(String(255), primary_key=True)  # Unique name of the AI model
    NumberParameters = Column(Float, nullable=False)  # Number of parameters in the model
    Fee = Column(Float, nullable=False)  # Fee to use this model
    Task = Column(String(255), nullable=False)  # Task description (e.g., classification, regression)

    records = relationship("Record", back_populates="model")
    tracking = relationship("Tracking", back_populates="model")

class Dataset(Base):
    __tablename__ = 'Dataset'
    
    Name = Column(String(255), primary_key=True)  # Unique dataset name
    Description = Column(Text)  # Description of the dataset

    records = relationship("Record", back_populates="dataset")

class Record(Base):
    __tablename__ = 'Record'
    
    # Composite primary key
    ModelName = Column(String(255), ForeignKey('Model.ModelName'), primary_key=True, nullable=False)  # Reference to Model table
    DatasetName = Column(String(255), ForeignKey('Dataset.Name'), primary_key=True, nullable=False)  # Reference to Dataset table
    Time = Column(Integer, nullable=False)  # Time taken for execution
    Device = Column(String(255), nullable=False)  # Device used for execution
    Metric = Column(Float, nullable=False)  # Performance metric (now as float)
    Unit = Column(String(255), nullable=False)  # Unit of the metric
    model = relationship("Model", back_populates="records")
    dataset = relationship("Dataset", back_populates="records")

class Tracking(Base):
    __tablename__ = 'Tracking'
    
    # Composite primary key
    UserID = Column(String(255), ForeignKey('User.ID'), primary_key=True, nullable=False)  # Reference to User table
    ModelName = Column(String(255), ForeignKey('Model.ModelName'), primary_key=True, nullable=False)  # Reference to Model table
    Feedback = Column(String(255))  # Feedback from the user

    user = relationship("User", back_populates="tracking")
    model = relationship("Model", back_populates="tracking")

# Example of creating the engine and tables
#engine = create_engine('sqlite:///BioApp.db')  # Replace with your actual database URL
#Base.metadata.create_all(engine)
