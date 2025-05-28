from pathlib import Path
import matplotlib.pyplot as plt
import json


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

similarity_threshold = 1  # Expertise score difference threshold in percentage points
threshold_dir = results_dir / f'adaptive_rsa/{similarity_threshold}_percent'


# Load the lists of subjects that were matched based on the similarity threshold
with open(results_dir / f'{threshold_dir}/{similarity_threshold}pct_matched_participants.json', 'r') as f:
    matched_participants = json.load(f)

with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)


# Calculate subgroup sizes
participants = list(matched_participants.keys())
subgroup_sizes = [len(matched_participants[p]) for p in participants]

# Sort participants by expertise score (descending order)
participants_with_expertise = [(p, expertise_scores[p], len(matched_participants[p])) 
                               for p in participants if p in expertise_scores]
sorted_data = sorted(participants_with_expertise, key=lambda x: x[1], reverse=True)
sorted_participants = [x[0] for x in sorted_data]
sorted_sizes = [x[2] for x in sorted_data]


plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(sorted_participants)), sorted_sizes, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
plt.xlabel('Participants (sorted by expertise score)', fontsize=12, fontweight='bold')
plt.ylabel('Number of Participants in Subgroup', fontsize=12, fontweight='bold')
plt.title(f'Subgroup Sizes\n(+- {similarity_threshold}% Expertise Similarity Threshold)', fontsize=14, fontweight='bold')
plt.xticks(range(len(sorted_participants)), sorted_participants, rotation=45, ha='right')
plt.yticks(range(0, max(sorted_sizes) + 2))

for i, participant in enumerate(sorted_participants):
    expertise_score = expertise_scores[participant]
    plt.text(i, sorted_sizes[i] + 0.1, f'{expertise_score:.1f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.grid(True, alpha=0.3, linestyle='-')
plt.tight_layout()
plt.savefig(results_dir / f'{threshold_dir}/{similarity_threshold}pct_subgroup__size_distribution.png', dpi=300, bbox_inches='tight')
