from Config import Config
import numpy as np
from model.randomforest import RandomForest

def model_predict(data, df, name):
    print(f"Model: {'randomforest'}")

    predictions_dict = {}
    accuracy_dict = {}
    intermediate_predictions = []

    for idx, class_col in enumerate(Config.TYPE_COLS):
        # Prepare features and labels
        if idx == 0:
            features = data.get_embeddings()
        else:
            features = np.column_stack(intermediate_predictions)

        # Initialize and train the model for the current class column
        model = RandomForest("RandomForest", features, class_col)
        model.train(data)
        model.predict(data)

        # Get predictions and true labels
        predictions = model.predictions
        y_test = data.get_type_y_test(class_col)

        # Get class mapping if available in data or model
        class_labels = np.unique(np.concatenate([predictions, y_test]))

        # Convert predictions and true labels to integer format
        label_to_int = {label: i for i, label in enumerate(class_labels)}
        predictions_int = [label_to_int.get(p, -1) for p in predictions]
        y_test_int = [label_to_int.get(t, -1) for t in y_test]

        # Compute accuracy
        accuracy = (np.array(predictions_int) == np.array(y_test_int)).mean() * 100

        # Store results in dictionaries and update intermediate predictions
        predictions_dict[class_col] = predictions_int
        accuracy_dict[class_col] = accuracy
        intermediate_predictions.append(predictions_int)

    # Calculate average accuracy for each group
    group_accuracies = {}
    group_values = np.unique(data.get_type(Config.GROUPED))

    for group_name in group_values:
        group_columns = [col for col in Config.TYPE_COLS if np.any(data.get_type(Config.GROUPED) == group_name)]
        group_acc = [accuracy_dict[col] for col in group_columns if col in accuracy_dict]
        if group_acc:
            avg_group_accuracy = np.mean(group_acc)
            group_accuracies[group_name] = avg_group_accuracy

    # Calculate overall average accuracy
    all_accuracies = list(accuracy_dict.values())
    if all_accuracies:
        overall_avg_accuracy = np.mean(all_accuracies)
    else:
            return acc
        
    print(f"Overall Average Accuracy for all groups: {overall_avg_accuracy:.2f}%")

    return predictions_dict, accuracy_dict, group_accuracies, overall_avg_accuracy 