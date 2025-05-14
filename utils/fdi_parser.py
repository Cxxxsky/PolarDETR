import os
import xml.etree.ElementTree as ET
import torch

class FDIParser:
    """
    Utility for parsing FDI tooth notation from FDI_MATCH.xml
    
    The FDI notation (ISO 3950) is a two-digit system where:
    - First digit: quadrant (1-4)
    - Second digit: tooth position in quadrant (1-8)
    """
    
    def __init__(self, xml_path=None):
        """
        Initialize FDI parser
        
        Args:
            xml_path (str, optional): Path to FDI_MATCH.xml file
        """
        self.xml_path = xml_path
        self.fdi_map = {}
        self.angle_map = {}
        
        # Default FDI to angle mapping if XML is not available
        self.default_angle_map = {
            # Upper right quadrant (1)
            11: 0, 12: 10, 13: 20, 14: 30, 15: 40, 16: 50, 17: 60, 18: 70,
            # Upper left quadrant (2)
            21: 180, 22: 170, 23: 160, 24: 150, 25: 140, 26: 130, 27: 120, 28: 110,
            # Lower left quadrant (3)
            31: 180, 32: 190, 33: 200, 34: 210, 35: 220, 36: 230, 37: 240, 38: 250,
            # Lower right quadrant (4)
            41: 0, 42: 350, 43: 340, 44: 330, 45: 320, 46: 310, 47: 300, 48: 290
        }
        
        # Default direction angle adjustments
        self.direction_angle_map = {
            "mesial": 15,
            "distal": -15,
            "buccal": 90,
            "lingual": -90,
            "labial": 90,
            "palatal": -90,
            "apical": 0,
            "coronal": 180
        }
        
        # Initialize with default map
        self.angle_map = self.default_angle_map.copy()
        
        # Try to load from XML if provided
        if xml_path and os.path.exists(xml_path):
            self.load_from_xml(xml_path)
    
    def load_from_xml(self, xml_path):
        """
        Load FDI tooth position mapping from XML file
        
        Args:
            xml_path (str): Path to FDI_MATCH.xml file
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Parse tooth positions
            for tooth in root.findall('./tooth'):
                fdi_number = int(tooth.get('fdi', 0))
                angle = float(tooth.get('angle', 0))
                description = tooth.get('description', '')
                
                if fdi_number > 0:
                    self.fdi_map[fdi_number] = {
                        'angle': angle,
                        'description': description
                    }
                    
                    # Update angle map
                    self.angle_map[fdi_number] = angle
            
            # Parse direction adjustments if present
            for direction in root.findall('./direction'):
                name = direction.get('name', '')
                angle = float(direction.get('angle', 0))
                
                if name:
                    self.direction_angle_map[name.lower()] = angle
                
            print(f"Loaded {len(self.fdi_map)} tooth positions from {xml_path}")
            
        except Exception as e:
            print(f"Error loading FDI_MATCH.xml: {e}")
            print("Using default FDI to angle mapping")
            self.angle_map = self.default_angle_map.copy()
    
    def get_angle(self, fdi_number):
        """
        Get angle for FDI tooth number
        
        Args:
            fdi_number (int): FDI tooth number
            
        Returns:
            float: Angle in degrees
        """
        return self.angle_map.get(fdi_number, 0)
    
    def get_direction_angle(self, direction):
        """
        Get angle adjustment for direction
        
        Args:
            direction (str): Direction name (e.g., "mesial", "distal")
            
        Returns:
            float: Angle adjustment in degrees
        """
        return self.direction_angle_map.get(direction.lower(), 0)
    
    def get_quadrant_angle(self, quadrant):
        """
        Get base angle for quadrant
        
        Args:
            quadrant (str): Quadrant identifier (1, 2, 3, 4, "UR", "UL", "LL", "LR")
            
        Returns:
            float: Base angle in degrees
        """
        quadrant_map = {
            "1": 0,      # Upper right
            "2": 180,    # Upper left
            "3": 180,    # Lower left
            "4": 0,      # Lower right
            "UR": 0,     # Upper right
            "UL": 180,   # Upper left
            "LL": 180,   # Lower left
            "LR": 0,     # Lower right
            "upper right": 0,
            "upper left": 180,
            "lower left": 180,
            "lower right": 0
        }
        
        # Convert numeric to string if needed
        if isinstance(quadrant, int):
            quadrant = str(quadrant)
            
        return quadrant_map.get(quadrant, 0)
    
    # The following methods are kept for backward compatibility but are not used 
    # with the BioClinicalBERT approach in the PTPE module
    
    def parse_fdi_from_text(self, text):
        """
        Extract FDI tooth number from text description (legacy method)
        
        Args:
            text (str): Text description
            
        Returns:
            int or None: FDI tooth number if found, None otherwise
        """
        # Try to match FDI notation directly if the BioClinicalBERT approach is not used!!!
        #  (e.g., "tooth 36" or "#36")
        import re
        fdi_pattern = r'(?:tooth|#)\s*(\d{2})'
        fdi_match = re.search(fdi_pattern, text.lower())
        
        if fdi_match:
            return int(fdi_match.group(1))
        
        # Try to match quadrant and position (e.g., "lower left second molar")
        quadrant_pattern = r'(upper|lower)\s+(right|left)'
        position_pattern = r'(first|second|third|1st|2nd|3rd)\s+(incisor|canine|premolar|molar)'
        
        quadrant_match = re.search(quadrant_pattern, text.lower())
        position_match = re.search(position_pattern, text.lower())
        
        if quadrant_match and position_match:
            quadrant_text = quadrant_match.group(1) + " " + quadrant_match.group(2)
            position_text = position_match.group(1) + " " + position_match.group(2)
            
            # Map quadrant text to number
            quadrant_map = {
                "upper right": 1,
                "upper left": 2,
                "lower left": 3,
                "lower right": 4
            }
            
            # Map position text to number
            tooth_map = {
                "first incisor": 1, "1st incisor": 1,
                "second incisor": 2, "2nd incisor": 2,
                "canine": 3,
                "first premolar": 4, "1st premolar": 4,
                "second premolar": 5, "2nd premolar": 5,
                "first molar": 6, "1st molar": 6,
                "second molar": 7, "2nd molar": 7,
                "third molar": 8, "3rd molar": 8
            }
            
            quadrant_num = quadrant_map.get(quadrant_text)
            tooth_num = tooth_map.get(position_text)
            
            if quadrant_num and tooth_num:
                return int(f"{quadrant_num}{tooth_num}")
        
        return None
    
    def extract_entities(self, text):
        """
        Legacy method for extracting anatomical entities from English text description
        Note: This method is maintained for backward compatibility but is not used with
        the BioClinicalBERT approach in the PTPE module.
        
        Args:
            text (str): Description text like "3mm cyst distal to tooth 37"
            
        Returns:
            dict: Dictionary with extracted entities
        """
        entities = {
            "quadrant": None,
            "tooth_number": None,
            "distance": None,
            "direction": None
        }
        
        # Extract FDI tooth number
        fdi_number = self.parse_fdi_from_text(text)
        if fdi_number:
            entities["tooth_number"] = fdi_number
            
            # Extract quadrant from FDI
            quadrant = str(fdi_number)[0]
            entities["quadrant"] = quadrant
        
        # Extract quadrant if not already found
        if not entities["quadrant"]:
            import re
            quadrant_pattern = r'(upper|lower)\s+(right|left)'
            quadrant_match = re.search(quadrant_pattern, text.lower())
            
            if quadrant_match:
                quadrant_text = quadrant_match.group(1) + " " + quadrant_match.group(2)
                
                # Map to quadrant codes
                quadrant_map = {
                    "upper right": "1",
                    "upper left": "2",
                    "lower left": "3",
                    "lower right": "4"
                }
                
                entities["quadrant"] = quadrant_map.get(quadrant_text)
        
        # Extract distance
        import re
        distance_pattern = r'(\d+\.?\d*)\s*(mm|cm)'
        distance_match = re.search(distance_pattern, text.lower())
        
        if distance_match:
            distance_value = float(distance_match.group(1))
            distance_unit = distance_match.group(2)
            
            # Convert to mm
            if distance_unit == "cm":
                distance_value *= 10
                
            entities["distance"] = distance_value
        else:
            # Default distance if not specified
            entities["distance"] = 5.0  # 5mm
        
        # Extract direction
        direction_words = ["mesial", "distal", "buccal", "lingual", "labial", "palatal", "apical", "coronal"]
        
        for direction in direction_words:
            if direction in text.lower():
                entities["direction"] = direction
                break
        
        return entities 