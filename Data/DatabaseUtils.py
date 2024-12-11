from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
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
databasemanager = DatabaseManager()
model_name = "LSTM-MOBILENETV3"
dataset=['ImageNet','ImageSmiles']
metrics=[75.2,0.02]
device="GPU"
time=  16
unit=["Accuracy","Loss"]


#usersID=databasemanager.getAlluserID()
#model_name=databasemanager.getAllModelName()



feedbacks = [
    "The model delivers highly accurate predictions consistently.",
    "The model occasionally takes too long to process large datasets.",
    "The model's outputs are stable and reproducible across runs.",
    "The model struggles with maintaining accuracy on noisy inputs.",
    "The model processes data efficiently, even under time constraints.",
    "The model's runtime performance could be improved for larger scales.",
    "The model demonstrates strong stability in multi-task scenarios.",
    "The model occasionally produces inconsistent results for similar inputs.",
    "The model offers an excellent balance of speed and accuracy.",
    "The model sometimes requires excessive computational resources, slowing down workflows.",
    "The model's stability makes it reliable for long-term applications.",
    "The model occasionally fails to converge during training on diverse datasets.",
    "The model's training time is manageable even with large datasets.",
    "The model's predictions are highly accurate but lack stability in certain edge cases.",
    "The model demonstrates robust performance under varying conditions.",
    "The model's inference time is slower compared to other state-of-the-art methods.",
    "The model maintains accuracy even with minimal parameter tuning.",
    "The model's predictions can be unstable when applied to unseen data.",
    "The model's speed makes it suitable for real-time applications.",
    "The model struggles with overfitting, requiring frequent monitoring.",
    "The model's training is efficient, requiring less time compared to alternatives.",
    "The model occasionally produces outputs with marginal accuracy drops.",
    "The model is stable under different initialization conditions.",
    "The model's performance degrades when trained on small datasets.",
    "The model achieves high accuracy with minimal pretraining.",
    "The model's runtime performance could benefit from optimization strategies.",
    "The model shows consistent accuracy across various benchmarks.",
    "The model's predictions occasionally lack reproducibility in complex scenarios.",
    "The model balances stability and computational efficiency effectively.",
    "The model's accuracy sometimes dips with highly imbalanced datasets.",
    "The model's training is fast but may sacrifice some predictive accuracy.",
    "The model demonstrates impressive stability during transfer learning tasks.",
    "The model's inference time could be improved for real-world deployment.",
    "The model's accuracy holds up well under cross-validation testing.",
    "The model struggles to generalize across significantly diverse data types.",
    "The model is highly stable, even during hyperparameter adjustments.",
    "The model requires longer training times compared to simpler architectures.",
    "The model achieves reliable accuracy without significant overfitting.",
    "The model's predictions occasionally vary depending on training epochs.",
    "The model demonstrates strong accuracy but requires stable infrastructure for deployment.",
    "The model's efficiency is ideal for time-sensitive applications.",
    "The model occasionally struggles with stability on smaller hardware configurations.",
    "The model is well-suited for tasks where accuracy is paramount.",
    "The model's convergence time is longer compared to baseline approaches.",
    "The model's stability ensures dependable predictions for mission-critical tasks.",
    "The model sometimes fails to deliver consistent accuracy under domain shifts.",
    "The model trains quickly without compromising prediction quality.",
    "The model exhibits slight instability when exposed to noisy training data.",
    "The model delivers near-instantaneous results, making it ideal for production.",
    "The model occasionally sacrifices accuracy for faster runtime performance.",
    "The model's stability allows it to integrate seamlessly into larger systems.",
    "The model takes longer to stabilize on datasets with high variability.",
    "The model ensures precision and accuracy across repeated experiments.",
    "The model's output consistency can vary with different random seeds.",
    "The model is fast and stable, making it ideal for iterative testing.",
    "The model's runtime on high-dimensional data is occasionally a bottleneck.",
    "The model's accuracy makes it a top choice for critical applications.",
    "The model exhibits instability during training on highly imbalanced datasets.",
    "The model processes data quickly, maintaining a balance with accuracy.",
    "The model's predictions are reliable but can be inconsistent with minor input changes.",
    "The model's stable performance enhances its adaptability to various tasks.",
    "The model's training time is higher but justified by its accuracy improvements.",
    "The model demonstrates robust accuracy but occasionally needs manual fine-tuning.",
    "The model maintains high stability across different operational environments.",
    "The model's speed is impressive but could trade off some prediction accuracy.",
    "The model occasionally fails to stabilize with non-standard input formats.",
    "The model’s high accuracy compensates for slightly longer inference times.",
    "The model's instability under specific configurations can hinder deployment.",
    "The model’s predictions are both fast and reliable, ensuring usability in real-time applications.",
    "The model requires extensive training time but delivers unmatched stability.",
    "The model's inconsistent accuracy in edge cases requires further investigation.",
    "The model’s predictions are remarkably stable across multiple hardware setups.",
    "The model delivers predictions quickly, making it suitable for real-time applications.",
    "The model requires significant processing time on larger datasets, which could be a limitation.",
    "The model maintains a good balance between speed and accuracy for most tasks.",
    "The model occasionally exhibits instability during training, especially on noisy data.",
    "The model achieves high accuracy in most benchmark tests, making it reliable for critical tasks.",
    "The model struggles to maintain accuracy when applied to out-of-distribution data.",
    "The model's stability across multiple runs ensures consistent performance for repeated experiments.",
    "The model experiences fluctuations in performance, indicating potential sensitivity to hyperparameter changes.",
    "The model's prediction time is relatively low, allowing for high-throughput analyses.",
    "The model's inference time could be optimized further for better efficiency in large-scale applications.",
    "The model consistently achieves high accuracy across diverse datasets, showcasing its robustness.",
    "The model's accuracy decreases slightly when handling highly imbalanced datasets.",
    "The model handles multitasking efficiently, minimizing time without compromising output quality.",
    "The model occasionally fails to converge, requiring additional tweaks to stabilize training.",
    "The model exhibits fast convergence during training, saving computational resources.",
    "The model's predictions are often delayed when processing high-dimensional data.",
    "The model's architecture provides a good trade-off between computational cost and prediction accuracy.",
    "The model needs improved stability mechanisms to avoid training interruptions in certain configurations.",
    "The model's accuracy aligns closely with human expert evaluations, building trust in its predictions.",
    "The model's training process takes more time compared to other similar models, which could be optimized.",
    "The model reliably produces results within an acceptable time frame for most applications.",
    "The model's instability in specific edge cases suggests a need for better regularization techniques.",
    "The model effectively balances speed, accuracy, and resource usage, making it versatile.",
    "The model requires better calibration for consistent accuracy across different input conditions.",
    "The model's stability during extended training sessions is a notable advantage.",
    "The model takes longer to process large inputs, which could impact its real-world usability.",
    "The model's predictions are quick and accurate, making it a preferred choice for rapid decision-making.",
    "The model demonstrates occasional inaccuracies when handling rare or unseen data points.",
    "The model's stable performance ensures reliable results even with challenging input distributions.",
    "The model's prediction time is competitive but could be improved for applications needing near-instant outputs.",
    "The model consistently outputs accurate predictions with minimal variance across repeated trials.",
    "The model sometimes struggles to stabilize during the initial training epochs.",
    "The model demonstrates impressive speed and accuracy, outperforming many other solutions.",
    "The model's accuracy is satisfactory for most tasks, though not groundbreaking.",
    "The model's inference process is highly stable, even with complex data pipelines.",
    "The model's training time is longer than expected, which may hinder iterative experimentation.",
    "The model's stability in high-dimensional spaces ensures confidence in its predictions.",
    "The model processes data efficiently but occasionally compromises accuracy for speed.",
    "The model strikes a solid balance between training stability and convergence speed.",
    "The model exhibits occasional instability when hyperparameters are not carefully tuned.",
    "The model produces predictions with high accuracy, though its runtime is above average.",
    "The model's accuracy exceeds expectations, but its training process is resource-intensive.",
    "The model performs well overall but requires more optimization for large-scale tasks.",
    "The model's consistent performance under varying conditions is a significant strength.",
    "The model experiences slow training times but compensates with high predictive accuracy.",
    "The model's efficient training pipeline ensures stable results with fewer epochs.",
    "The model's slower runtime could limit its adoption for time-critical applications.",
    "The model's reliability in delivering accurate predictions within a stable framework is commendable.",
    "The model sometimes sacrifices speed for achieving greater accuracy.",
    "The model's training stability is sufficient for most standard configurations.",
    "The model's slow convergence rate suggests potential room for architectural improvements.",
    "The model's predictions are stable across multiple datasets, showcasing its generalizability.",
    "The model's computation time increases significantly as dataset size grows.",
    "The model's accuracy remains consistent over multiple tasks, building trust in its utility.",
    "The model requires additional training to stabilize its performance on edge cases.",
    "The model's fast runtime ensures it can scale effectively for large datasets.",
    "The model's training process exhibits minor instability on high-noise datasets.",
    "The model's accuracy aligns well with human expert evaluations, providing confidence in its predictions.",
    "The model's training stability is a significant advantage for iterative experimentation.",
    "The model's accuracy is competitive, but not groundbreaking.",
    "The model's training time is longer than expected, which may hinder iterative experimentation.",
    "The model's stability in high-dimensional spaces ensures confidence in its predictions.",
    "The model processes data efficiently but occasionally compromises accuracy for speed.",
    "The model strikes a solid balance between training stability and convergence speed.",
    "The model exhibits occasional instability when hyperparameters are not carefully tuned.",
    "The model produces predictions with high accuracy, though its runtime is above average.",
    "The model's accuracy exceeds expectations, but its training process is resource-intensive.",
    "The model performs well overall but requires more optimization for large-scale tasks.",
    "The model's consistent performance under varying conditions is a significant strength.",
    "The model experiences slow training times but compensates with high predictive accuracy.",
    "The model's efficient training pipeline ensures stable results with fewer epochs.",
      "The model achieves excellent trade-offs between training time and prediction quality.",
    "The model's performance degrades slightly when scaling to larger datasets.",
    "The model produces highly accurate predictions with a minimal margin of error.",
    "The model takes longer to train compared to other models of similar complexity.",
    "The model is stable under a wide range of parameter configurations, making it reliable.",
    "The model sometimes experiences instability when dealing with noisy or incomplete data.",
    "The model's fast training times make it ideal for iterative development processes.",
    "The model's accuracy is exceptional, especially for tasks with clear feature representation.",
    "The model struggles to maintain high accuracy for tasks involving complex feature interactions.",
    "The model's runtime efficiency makes it suitable for high-throughput pipelines.",
    "The model's inference time is slightly higher than expected for real-time applications.",
    "The model is robust and handles diverse datasets without significant performance loss.",
    "The model exhibits occasional instability during hyperparameter optimization.",
    "The model's training time is reasonable, even for large-scale datasets.",
    "The model requires better initialization techniques to stabilize early training phases.",
    "The model achieves consistent results across various cross-validation splits.",
    "The model's slower inference speed could limit its practicality in some scenarios.",
    "The model excels at tasks requiring high accuracy and fine-grained predictions.",
    "The model's stability across iterations builds confidence in its reliability.",
    "The model requires significant computational resources, impacting deployment on standard hardware.",
    "The model handles data with high dimensionality efficiently, maintaining stability.",
    "The model's slower convergence in training could be optimized for better usability.",
    "The model produces consistent outputs, reducing the need for repeated verifications.",
    "The model occasionally fails to stabilize during the final stages of training.",
    "The model's predictions are highly accurate for most applications, but runtime can be improved.",
    "The model's runtime efficiency ensures compatibility with real-time requirements.",
    "The model's stability under varying batch sizes is an advantage in flexible workflows.",
    "The model struggles to maintain stable predictions when data preprocessing is inconsistent.",
    "The model demonstrates excellent speed-to-accuracy ratios, even with limited resources.",
    "The model requires additional tuning to address instability on edge-case scenarios.",
    "The model's generalizability is enhanced by its stable performance across diverse datasets.",
    "The model sometimes requires extended training to achieve peak accuracy levels.",
    "The model's ability to produce consistent predictions improves its overall utility.",
    "The model's performance is affected by minor instability in the middle training epochs.",
    "The model's fast convergence reduces overall experimentation time.",
    "The model struggles with runtime efficiency when processing large batch sizes.",
    "The model's accuracy in benchmark tests highlights its robustness against noise.",
    "The model experiences slight instability when dealing with overlapping features.",
    "The model's high accuracy compensates for its moderately high computation time.",
    "The model's stable performance across different initializations ensures reliability.",
    "The model's slower runtime for predictions may limit its real-world application in time-sensitive tasks.",
    "The model consistently delivers accurate outputs, even under constrained resource conditions.",
    "The model's stability during inference contributes to its reproducibility.",
    "The model's training instability under specific configurations suggests room for improvement.",
    "The model's efficient use of computational resources enhances scalability.",
    "The model occasionally requires retraining to address drift in input distributions.",
    "The model's runtime scales predictably with dataset size, ensuring transparency in deployment.",
    "The model struggles to maintain accuracy on datasets with extreme variability.",
    "The model's predictions are highly reproducible, showcasing its stable architecture.",
    "The model requires additional computational power to optimize its accuracy-speed trade-offs.",
    "The model performs well across a broad range of accuracy metrics, ensuring reliability.",
    "The model sometimes experiences instability when trained on imbalanced datasets.",
    "The model's training speed is satisfactory for prototyping and iterative development.",
    "The model requires better parameter tuning to handle complex feature interactions efficiently.",
    "The model demonstrates stable and predictable behavior across unseen test cases.",
    "The model's convergence speed is slower compared to some state-of-the-art alternatives.",
    "The model ensures accurate predictions without excessive computational overhead.",
    "The model's runtime efficiency makes it a good candidate for scalable workflows.",
    "The model sometimes encounters instability during transfer learning applications.",
    "The model's performance on accuracy benchmarks is competitive, though training time could be improved.",
    "The model's stability across different datasets makes it a versatile tool.",
    "The model requires additional debugging to address rare instability issues.",
    "The model's fast runtime enables seamless integration into real-time systems.",
    "The model occasionally shows slight drops in accuracy when scaling to unseen datasets.",
    "The model's consistent accuracy across tasks ensures reliable predictions in production.",
    "The model's initialization stability enhances its reliability during experimentation.",
     "The model achieves high accuracy in most scenarios but struggles with edge cases.",
    "The model's inference time is optimized for real-time applications.",
    "The model demonstrates occasional instability in highly variable datasets.",
    "The model delivers consistently accurate results across diverse datasets.",
    "The model's training time is significant, limiting rapid prototyping.",
    "The model's predictions are reliable and maintain a high level of precision.",
    "The model sometimes exhibits instability when processing noisy inputs.",
    "The model is computationally efficient and well-suited for large-scale applications.",
    "The model's runtime performance occasionally hinders usability in production.",
    "The model offers stable results, even with minimal parameter adjustments.",
    "The model's accuracy degrades when applied to datasets outside its training domain.",
    "The model processes large datasets efficiently, reducing overall runtime.",
    "The model struggles to converge when trained on highly imbalanced datasets.",
    "The model maintains stable performance across varying input conditions.",
    "The model's runtime can be prohibitive for extremely high-dimensional data.",
    "The model achieves consistent accuracy with minimal manual intervention.",
    "The model's instability in certain configurations affects deployment reliability.",
    "The model's efficiency is ideal for applications requiring quick turnarounds.",
    "The model occasionally sacrifices stability for speed in complex scenarios.",
    "The model delivers accurate results, but longer training times are a drawback.",
    "The model exhibits strong stability, even under resource-constrained environments.",
    "The model's runtime can become a bottleneck with larger-than-expected datasets.",
    "The model consistently achieves high accuracy across benchmark datasets.",
    "The model requires additional fine-tuning to stabilize performance on unseen data.",
    "The model processes small datasets quickly, making it versatile for research.",
    "The model occasionally fails to maintain accuracy over repeated iterations.",
    "The model's stability allows for seamless integration into pipelines.",
    "The model's high runtime efficiency does not compromise its predictive power.",
    "The model exhibits slight instability in highly dynamic input scenarios.",
    "The model's training pipeline is straightforward but time-intensive.",
    "The model delivers reliable accuracy across varied operational conditions.",
    "The model occasionally underperforms when generalizing to new tasks.",
    "The model processes inputs quickly, offering a balance between speed and precision.",
    "The model's performance can be inconsistent with small training datasets.",
    "The model maintains stability across different hardware configurations.",
    "The model's training time could be optimized for quicker iterations.",
    "The model achieves near-perfect accuracy in controlled experimental setups.",
    "The model occasionally struggles with stability during cross-validation.",
    "The model processes large-scale inputs rapidly without losing accuracy.",
    "The model's predictions lack consistency in highly specialized domains.",
    "The model's runtime is manageable, even for computationally intensive tasks.",
    "The model's accuracy occasionally drops on long-tail distribution data.",
    "The model shows stable behavior when applied to diverse use cases.",
    "The model requires significant computational resources to maintain its speed.",
    "The model demonstrates excellent stability in multi-task learning environments.",
    "The model struggles to deliver accurate results under strict runtime constraints.",
    "The model's speed and stability make it suitable for industrial applications.",
    "The model's accuracy varies when tested across non-standard benchmarks.",
    "The model provides consistent runtime performance regardless of input size.",
    "The model's training process can be lengthy, affecting iterative workflows.",
    "The model's accuracy remains stable across multiple testing conditions.",
    "The model occasionally produces unstable outputs when retrained on new data.",
    "The model’s runtime efficiency allows for real-time deployment in production.",
    "The model’s stability falters with non-uniformly distributed training data.",
    "The model achieves an excellent balance of runtime efficiency and accuracy.",
    "The model occasionally fails to stabilize with minimal training epochs.",
    "The model demonstrates robust accuracy but requires careful runtime optimization.",
    "The model processes data quickly while maintaining consistent output quality.",
    "The model's predictions are reliable but can occasionally vary with initialization.",
    "The model handles complex inputs with remarkable stability and precision.",
    "The model's training duration could hinder adoption in time-sensitive projects.",
    "The model delivers high accuracy in testing but struggles during inference on new data.",
    "The model’s speed and accuracy make it a preferred choice for high-throughput tasks.",
    "The model occasionally produces inconsistent outputs across different runs.",
    "The model demonstrates stable behavior in both training and testing phases.",
    "The model's runtime can be optimized further for deployment in resource-limited environments.",
    "The model maintains high accuracy but could benefit from enhanced training stability.",
    "The model's quick inference time is ideal for scenarios requiring rapid decisions.",
    "The model's stability is compromised when trained on small, noisy datasets.",
    "The model consistently delivers precise predictions in controlled experiments.",
    "The model occasionally takes longer to process than comparable architectures.",
    "The model exhibits excellent runtime performance, making it production-ready.",
    "The model's training pipeline needs optimization for faster convergence.",
    "The model achieves remarkable stability with minimal hyperparameter adjustments.",
    "The model's accuracy fluctuates slightly with changes in input configurations.",
    "The model processes data efficiently without sacrificing prediction quality.",
    "The model struggles to stabilize when exposed to high-variability inputs.",
    "The model’s efficiency and stability make it suitable for critical applications.",
    "The model occasionally overfits, requiring adjustments during training.",
    "The model demonstrates reliable accuracy but slower runtime compared to competitors.",
    "The model consistently maintains stability, even with dynamic input conditions.",
    "The model's training time is reasonable but could be further streamlined.",
    "The model’s accuracy is high but requires frequent validation for stability.",
    "The model efficiently handles large datasets with minimal computational overhead.",
    "The model occasionally struggles to generalize across unseen data distributions.",
    "The model exhibits stability across varied operational conditions, improving trustworthiness."
]


#get tracking
        
#get record
#databasemanager.get_all_models()




#add Admin account
admin_id='admin123'
admin_password='admin123'
admin_email='nt1179771@gmail.com'
admin_fullname='Michael Miller'
#databasemanager.add_admin(admin_id,admin_fullname,admin_email,admin_password)
user_id='ttn1'
password='password123'
admin_email='ttn@gmail.com'
admin_fullname='Michael Richardson'
#databasemanager.add_user(user_id,password,admin_email,admin_fullname)








#add_record(self, model_name, dataset_name, time, device, metric, unit)
#for i in range(len(dataset)):
    #databasemanager.add_record(model_name,dataset[i],time,device,metrics[i],unit[i])


#records=databasemanager.get_record_by_model_name(model_name)
#if records:
    #for record in records:
        #print(f"Model: {record.ModelName}, Dataset: {record.DatasetName}, Time: {record.Time}, Device: {record.Device}, Metric: {record.Metric}, Unit: {record.Unit}")
#else:
    #print(f"No records found for model: {model_name}")