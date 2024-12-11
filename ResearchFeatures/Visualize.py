import os
import urllib.parse
from rdkit import Chem
from rdkit.Chem import Draw

class Visualize:
    def __init__(self):
        # Ensure a directory exists to store temporary images
        self.image_dir = "static/temp_images"
        os.makedirs(self.image_dir, exist_ok=True)

    def smileImage(self, smile):
        try:
            mol = Chem.MolFromSmiles(smile)
            if not mol:
                return None

            # Sanitize the SMILES string to create a valid filename
            img_filename = f"{smile}.png".replace("/", "_")

            # URL encode the filename to handle special characters like '#'
            encoded_filename = urllib.parse.quote(img_filename)

            # Define the image path
            img_path = os.path.join(self.image_dir, img_filename)
            print('save file',img_path)
            # Save the molecule image
            Draw.MolToFile(mol, img_path)
            # Return the URL for the static image
            return f"/static/temp_images/{encoded_filename}"
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
