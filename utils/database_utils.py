from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.Entities import *
import bcrypt

import random
class DatabaseManager:
    def __init__(self):
        self.engine = create_engine('sqlite:///BioApp.db')
        self.Session = sessionmaker(bind=self.engine)

    # 1. Function to add a new user
    def add_user(self, user_id, full_name, email, password, level, payment=0):
        try:
            with self.Session() as session:
                new_user = User(ID=user_id, FullName=full_name, Email=email, Password=password, Level=level, payment=payment)
                session.add(new_user)
                session.commit()
                print(f"User {full_name} added successfully!")
        except Exception as e:
            print(f"Error adding user: {e}")
    
    
    
    
    
    
    
    
    #2. Function to Get users by id
    def get_user_by_id(self, user_id):
        try:
            with self.Session() as session:
                user = session.query(User).filter(User.ID == user_id).first()
                if user:
                    return user
                else:
                    return None
        except Exception as e:
            print(f"Error fetching user: {e}")
    
    
    
    
    
    
    
    
    #3. Function to get all users
    def get_all_users(self):
        results=[]
        try:
            with self.Session() as session:
                users = session.query(User).all()
                for user in users:
                    data={"ID":user.ID,"FullName":user.FullName,"Email":user.Email,"Level":user.Level,"Password":user.Password,"Payment":user.payment}
                    results.append(data)
                    print(f"ID: {user.ID}, FullName: {user.FullName}, Email: {user.Email}, Level: {user.Level}, Password: {user.Password}, Payment: {user.payment}")
            return results
        except Exception as e:
            print(f"Error fetching users: {e}")
    

        
    
    
    
    
    
    
    
    
    #4. Function to delete a user by ID
    def delete_user(self, user_id):
        try:
            with self.Session() as session:
                user = session.query(User).filter(User.ID == user_id).first()
                if user:
                    session.delete(user)
                    session.commit()
                    print(f"User {user_id} deleted successfully!")
                else:
                    print(f"User {user_id} not found.")
        except Exception as e:
            print(f"Error deleting user: {e}")

    
    #update user payment
    def update_user_payment(self, user_id, new_payment):
     try:
        with self.Session() as session:
            # Fetch the user from the database
            user = session.query(User).filter(User.ID == user_id).first()
            if user:
                # Update the user's payment
                user.payment += new_payment
                # Commit the changes
                session.commit()
                print(f"User {user_id}'s payment updated to {new_payment}.")
            else:
                print(f"User {user_id} not found.")
     except Exception as e:
        print(f"Error updating user payment: {e}")
    

    #random initialize all users payment
    def random_initialize_user_payment(self):
     try:
        with self.Session() as session:
            # Fetch all users from the database
            users = session.query(User).all()
            for user in users:
                # Update the user's payment
                user.payment = random.randint(300, 1000)
                # Commit the changes
                session.commit()
                print(f"User {user.ID}'s payment updated to {user.payment}.")
     except Exception as e:
        print(f"Error updating user payment: {e}")  
    #reset all level of users to 1
    def reset_user_level(self):
     try:
        with self.Session() as session:
            # Fetch all users from the database
            users = session.query(User).all()
            for user in users:
                # Update the user's level
                user.Level = '1'
                # Commit the changes
                session.commit()
                print(f"User {user.ID}'s level reset to 1.")
     except Exception as e:
        print(f"Error resetting user level: {e}")
    
    #reset password
    def reset_user_password(self, user_id, new_password):
     try:
        with self.Session() as session:
            # Fetch the user from the database
            user = session.query(User).filter(User.ID == user_id).first()
            if user:
                # Update the user's password
                user.Password = new_password
                # Commit the changes
                session.commit()
                print(f"User {user_id}'s password reset.")
            else:
                print(f"User {user_id} not found.")
     except Exception as e:
        print(f"Error resetting user password: {e}")
    
    #reset all password
    def reset_all_password(self):
     try:
        with self.Session() as session:
            # Fetch all users from the database
            users = session.query(User).all()
            for user in users:
                # Update the user's password
                user.Password = 'password123'
                # Commit the changes
                session.commit()
                print(f"User {user.ID}'s password reset.")
     except Exception as e:
        print(f"Error resetting user password: {e}")

    #update user level
    def update_user_level(self, user_id, new_level):
     try:
        with self.Session() as session:
            # Fetch the user from the database
            user = session.query(User).filter(User.ID == user_id).first()
            if user:
                print(f"User {user_id} before update: payment = {user.payment}")  # Debugging print
                
                # Update the user's level and payment based on the new level
                user.Level = new_level
                if new_level == '2':
                    self.update_user_payment(user_id, 100)
                elif new_level == '3':
                    self.update_user_payment(user_id, 200)

                # Force the changes to be written to the database
                session.flush()  # Ensures changes are reflected immediately
                
                print(f"User {user_id} after update: payment = {user.payment}")  # Debugging print
                
                # Commit the changes
                session.commit()
                print(f"User {user_id}'s level updated to {new_level}.")
            else:
                print(f"User {user_id} not found.")
     except Exception as e:
        print(f"Error updating user level: {e}")

    #get all userID
    def getAlluserID(self):
        try:
            results=[]
            with self.Session() as session:
                users = session.query(User).all()
                for user in users:
                    results.append(user.ID)
                    print(f"ID: {user.ID}, FullName: {user.FullName}, Email: {user.Email}, Level: {user.Level}, Password: {user.Password}, Payment: {user.payment}")
                return results
        except Exception as e:    
            print(f"Error fetching users: {e}")
    
    
    #change password
    def change_password(self, user_id, new_password):
        try:
            with self.Session() as session:
                user = session.query(User).filter(User.ID == user_id).first()
                if user:
                    user.Password = new_password
                    session.commit()
                    print(f"User {user_id}'s password changed to {new_password}.")
                else:
                    print(f"User {user_id} not found.")
        except Exception as e:
            print(f"Error changing password: {e}")

    #change email
    def change_email(self, user_id, new_email):
        try:
            with self.Session() as session:
                user = session.query(User).filter(User.ID == user_id).first()
                if user:
                    user.Email = new_email
                    session.commit()
                    print(f"User {user_id}'s email changed to {new_email}.")
                else:
                    print(f"User {user_id} not found.")
        except Exception as e:
            print(f"Error changing email: {e}")


    #5. Function to add a new admin
    def add_admin(self, admin_id, full_name, email, password):
        try:
            with self.Session() as session:
                new_admin = Admin(ID=admin_id, FullName=full_name, Email=email, Password=password)
                session.add(new_admin)
                session.commit()
                print(f"Admin {full_name} added successfully!")
        except Exception as e:
            print(f"Error adding admin: {e}")

    #get admin
    def get_admin(self, admin_id):
        try:    
            with self.Session() as session:
                admin = session.query(Admin).filter(Admin.ID == admin_id).first()   
                if admin:
                    return admin
                else:
                    return None
        except Exception as e:
            print(f"Error fetching admin: {e}")
    
    
    
    
    
    
    
    #6. Function to add a new model
    def add_model(self, model_name, num_parameters, fee, task):
        try:
            with self.Session() as session:
                new_model = Model(ModelName=model_name, NumberParameters=num_parameters, Fee=fee, Task=task)
                session.add(new_model)
                session.commit()
                print(f"Model {model_name} added successfully!")
        except Exception as e:
            print(f"Error adding model: {e}")

    
    
    
    
    
    
    #7. Function to get a model by name
    def get_model(self, model_name):
        try:
            with self.Session() as session:
                model = session.query(Model).filter(Model.ModelName == model_name).first()
                if model:
                    print(f"Model: {model.ModelName}, Parameters: {model.NumberParameters}, Fee: {model.Fee}, Task: {model.Task}")
                else:
                    print(f"Model {model_name} not found.")
        except Exception as e:
            print(f"Error fetching model: {e}")

        return model
    
    
    
    
    
    
    #8. Function to get all models
    def get_all_models(self):
        results=[]
        try:
            with self.Session() as session:
                models = session.query(Model).all()
                for model in models:
                    data={"ModelName":model.ModelName,"NumberParameters":model.NumberParameters,"Fee":model.Fee,"Task":model.Task}
                    results.append(data)
                    print(f"Model: {model.ModelName}, Parameters: {model.NumberParameters}, Fee: {model.Fee}, Task: {model.Task}")
            return results
        except Exception as e:
            print(f"Error fetching models: {e}")
    
    
    
    
    #get all model name
    def getAllModelName(self):
        results=[]
        try:
            with self.Session() as session:
                models = session.query(Model).all()
                for model in models:
                    results.append(model.ModelName)
                    print(f"Model: {model.ModelName}")
            return results
        except Exception as e:
            print(f"Error fetching models: {e}")
    
    
    
    9#Function to delete a model by name (with dependency check)
    def delete_model(self, model_name):
     try:
        with self.Session() as session:
            # Fetch all dependent records in Tracking and Record tables
            dependent_tracking = session.query(Tracking).filter(Tracking.ModelName == model_name).all()
            dependent_records = session.query(Record).filter(Record.ModelName == model_name).all()
            
            # Delete the dependent tracking and record entries
            for track in dependent_tracking:
                session.delete(track)
                print(f"Deleted tracking for model {model_name} and user {track.UserID}")
            
            for record in dependent_records:
                session.delete(record)
                print(f"Deleted record for model {model_name} and dataset {record.DatasetName}")

            # Now delete the model itself
            model = session.query(Model).filter(Model.ModelName == model_name).first()
            if model:
                session.delete(model)
                session.commit()
                print(f"Model {model_name} and all its related records deleted successfully!")
            else:
                print(f"Model {model_name} not found.")
     except Exception as e:
        print(f"Error deleting model: {e}")


    
    
    
    
    #10. Function to add a new dataset
    def add_dataset(self, dataset_name, description):
        try:
            with self.Session() as session:
                new_dataset = Dataset(Name=dataset_name, Description=description)
                session.add(new_dataset)
                session.commit()
                print(f"Dataset {dataset_name} added successfully!")
        except Exception as e:
            print(f"Error adding dataset: {e}")

    
    
    
    
    
    #11. Function to get all datasets
    def get_all_datasets(self):
        results=[]
        try:
            with self.Session() as session:
                datasets = session.query(Dataset).all()
                for dataset in datasets:
                    data={"Name":dataset.Name,"Description":dataset.Description}
                    results.append(data)
                    print(f"Name: {dataset.Name}, Description: {dataset.Description}")
            return results
        except Exception as e:
            print(f"Error fetching datasets: {e}")
    
    


    
    
    
    
    # 12. Function to delete a dataset by name (with dependency handling)
    def delete_dataset(self, dataset_name):
        try:
            with self.Session() as session:
                # Find all records that depend on the dataset
                dependent_records = session.query(Record).filter(Record.DatasetName == dataset_name).all()
                
                # Delete dependent records first
                for record in dependent_records:
                    session.delete(record)
                    print(f"Deleted record for dataset {dataset_name} and model {record.ModelName}")

                # Now delete the dataset itself
                dataset = session.query(Dataset).filter(Dataset.Name == dataset_name).first()
                if dataset:
                    session.delete(dataset)
                    session.commit()
                    print(f"Dataset {dataset_name} and all its dependent records deleted successfully!")
                else:
                    print(f"Dataset {dataset_name} not found.")
        except Exception as e:
            print(f"Error deleting dataset: {e}")
    
    
    
    # 13. Function to get a dataset by name
    def get_dataset(self, dataset_name):
        try:
            with self.Session() as session: 
                dataset = session.query(Dataset).filter(Dataset.Name == dataset_name).first()
                if dataset:
                    return dataset
                else:
                    return None
        except Exception as e:
            print(f"Error fetching dataset: {e}")
    
    
    
    
    #14. Function to add a new record
    def add_record(self, model_name, dataset_name, time, device, metric, unit):
        try:
            with self.Session() as session:
                new_record = Record(ModelName=model_name, DatasetName=dataset_name, Time=time, Device=device, Metric=metric, Unit=unit)
                session.add(new_record)
                session.commit()
                print(f"Record for Model: {model_name}, Dataset: {dataset_name} added successfully!")
        except Exception as e:
            print(f"Error adding record: {e}")
    

    #15 Function to get all records
    def get_all_records(self):
        results=[]
        try:
            with self.Session() as session:
                records = session.query(Record).all()
                for record in records:
                    data={"ModelName":record.ModelName,"DatasetName":record.DatasetName,"Time":record.Time,"Device":record.Device,"Metric":record.Metric,"Unit":record.Unit}
                    results.append(data)
                    print(f"Model: {record.ModelName}, Dataset: {record.DatasetName}, Time: {record.Time}, Device: {record.Device}, Metric: {record.Metric}, Unit: {record.Unit}")
            return results
        except Exception as e:
            print(f"Error fetching records: {e}")

    #16 function to get all records by model name
    def get_record_by_model_name(self, model_name):
        try:
            with self.Session() as session:
                records = session.query(Record).filter(Record.ModelName == model_name).all()
                if records:
                    return records
                for record in records:
                    print(f"Model: {record.ModelName}, Dataset: {record.DatasetName}, Time: {record.Time}, Device: {record.Device}, Metric: {record.Metric}, Unit: {record.Unit}")
                else:
                    print(f"No records found for model: {model_name}")
                    return None
        except Exception as e:
            print(f"Error fetching record by model name: {e}")
    
    #Update a record by model name
    def update_record_by_model_name(self, model_name, dataset_name, time, device, metric, unit):
        try:
            with self.Session() as session:
                record = session.query(Record).filter(Record.ModelName == model_name, Record.DatasetName == dataset_name).first()
                if record:
                    record.DatasetName = dataset_name
                    record.Time = time
                    record.Device = device
                    record.Metric = metric
                    record.Unit = unit
                    session.commit()
                    print(f"Record for Model: {model_name} updated successfully!")
                else:
                    print(f"No record found for model: {model_name}")    
        except Exception as e:
            print(f"Error updating record by model name: {e}")
    
    #delete a record by model name and dataset name
    def delete_record_by_model_dataset(self, model_name, dataset_name):
        try:    
            with self.Session() as session:
                record = session.query(Record).filter(Record.ModelName == model_name, Record.DatasetName == dataset_name).first()
                if record:
                    session.delete(record)
                    session.commit()
                    print(f"Record for Model: {model_name} and Dataset: {dataset_name} deleted successfully!")
                else:
                    print(f"No record found for model: {model_name} and dataset: {dataset_name}")
        except Exception as e:
            print(f"Error deleting record by model name and dataset name: {e}")
    
    
    
    # 17. Function to get all records by dataset name
    def get_record_by_dataset_name(self, dataset_name):
        try:
            with self.Session() as session:
                records = session.query(Record).filter(Record.DatasetName == dataset_name).all()  # Changed to .all()
                if records:
                    return records  # Return all records associated with the dataset
                else:
                    return []  # Return an empty list if no records are found
        except Exception as e:
            print(f"Error fetching records by dataset name: {e}")
            return []


    #18. Function to add tracking information
    def add_tracking(self, user_id, model_name, feedback):
        try:
            with self.Session() as session:
                new_tracking = Tracking(UserID=user_id, ModelName=model_name, Feedback=feedback)
                session.add(new_tracking)
                session.commit()
                print(f"Tracking for User: {user_id}, Model: {model_name} added successfully!")
        except Exception as e:
            print(f"Error adding tracking: {e}")

    #19. Function to get tracking information for a user
    def get_tracking_by_user(self, user_id):
        try:
            with self.Session() as session:
                tracking = session.query(Tracking).filter(Tracking.UserID == user_id).all()
                for track in tracking:
                    print(f"User: {track.UserID}, Model: {track.ModelName}, Feedback: {track.Feedback}")
        except Exception as e:
            print(f"Error fetching tracking information: {e}")
    # 20. Function to get all tracking information
    def getalltracking(self):
        try:
            results = []
            with self.Session() as session:
                tracking = session.query(Tracking).all()
                for track in tracking:
                    data={"UserID":track.UserID,"ModelName":track.ModelName,"Feedback":track.Feedback}
                    results.append(data)
                    print(f"UserID: {track.UserID}, ModelName: {track.ModelName}, Feedback: {track.Feedback}")
                return results
        except Exception as e:
            print(f"Error fetching tracking information: {e}")
    #20. Function to get tracking information for a model
    def get_tracking_by_model(self, model_name):
        result = []
        try:
            with self.Session() as session:
                tracking = session.query(Tracking).filter(Tracking.ModelName == model_name).all()
                for track in tracking:
                    data={"UserID":track.UserID,"ModelName":track.ModelName,"Feedback":track.Feedback}
                    result.append(data)
                    print(f"User: {track.UserID}, Model: {track.ModelName}, Feedback: {track.Feedback}")
                return result
        except Exception as e:
            print(f"Error fetching tracking information: {e}")
    #21 Function to get tracking information from a user
    def get_tracking_by_user_id(self, user_id):
        result = []
        try:
            with self.Session() as session:
                tracking = session.query(Tracking).filter(Tracking.UserID == user_id).all()
                for track in tracking:
                    data={"UserID":track.UserID,"ModelName":track.ModelName,"Feedback":track.Feedback}
                    result.append(data)
                    print(f"User: {track.UserID}, Model: {track.ModelName}, Feedback: {track.Feedback}")
                return result
        except Exception as e:
            print(f"Error fetching tracking information: {e}")   
    # function to change feedback of users 
    def change_feedback(self, user_id, model_name, new_feedback):
        try:
            with self.Session() as session: 
                #if not tracking exists add to it

                tracking = session.query(Tracking).filter(Tracking.UserID == user_id, Tracking.ModelName == model_name).first()
                if tracking:
                    tracking.Feedback = new_feedback
                    session.commit()
                    print(f"Feedback for User: {user_id}, Model: {model_name} changed to: {new_feedback}")
                else:
                    self.add_tracking(user_id, model_name, new_feedback)
                    print(f"Tracking for User: {user_id}, Model: {model_name} added successfully!")
        except Exception as e:
            print(f"Error changing feedback: {e}")