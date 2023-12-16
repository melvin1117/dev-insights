import json

def calculate_proficiency(entry):
    # Define scoring system
    beginner_score = 30
    intermediate_score = 42

    # Get values from the entry
    years_code = int(entry.get("YearsCode", 0))
    employment = bool(entry.get("Employment", 0))

    # Calculate scores based on coding experience
    coding_experience_score = 0
    if years_code <= 30:
        coding_experience_score = beginner_score
    elif 30 < years_code <= 42:
        coding_experience_score = intermediate_score
    else:
        coding_experience_score = intermediate_score + 1  # Adjust as needed

    # Calculate scores based on employment status
    employment_score = 0
    if employment == 1:
        employment_score = 2  # Adjust as needed

    # Combine scores
    combined_score = coding_experience_score + employment_score

    # Determine proficiency level
    proficiency_level = ""
    if combined_score <= beginner_score:
        proficiency_level = "Beginner"
    elif beginner_score < combined_score <= intermediate_score:
        proficiency_level = "Intermediate"
    else:
        proficiency_level = "Expert"

    return proficiency_level

def add_proficiency_level(input_file_path, output_file_path):
    # Read the input JSON file
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # Add proficiency level to each entry
    for entry in data:
        proficiency_level = calculate_proficiency(entry)
        entry["ProficiencyLevel"] = proficiency_level

    # Write the data with proficiency level to a new JSON file
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Example usage
input_file_path = './assets/so/survey_results_public.json'
output_file_path = './assets/output_final_result.json'
add_proficiency_level(input_file_path, output_file_path)