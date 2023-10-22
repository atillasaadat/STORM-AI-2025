# Metrics
The performance of the participant submissions will be evaluated based on their ability to accurately detect and classify satellite behavioral mode changes compared to the ground truth labels. The key metric used to create the leaderboard is the $F_2$ score, which is a variant of the $F_1$ score that emphasizes recall over precision. The precision corresponds to the proportion of detected nodes (True Positives (TPs) and False Positives (FPs)) that were correct, whereas the recall corresponds to the percentage of actual true nodes (True Positives (TPs) and False Negatives (FNs)) that were detected. Given that, the $F_2$ score is given as:

$$Precision = TP/(TP + FP)$$

$$Recall = TP/(TP + FN)$$

$$F_2 = 5(Precision \cdot Recall)/4(Precision+Recall) $$

The definition og true positive ($TP$), false positive ($FP$), and false negative ($FN$) are described in the table below:

| Acronym  | Description |
| ------------- | ------------- |
| TP  | The participant node falls within the time tolerance interval of the ground truth node and the labels match  |
| FP  | Two possibilities: (1) The participant node falls within the time tolerance interval of a ground truth node, but the labels do not match. (2) There are no ground truth nodes within the tolerance interval around the participant node.  |
| FN  | Missed ground truth node; there is no participant node within the tolerance interval of a ground truth node  |

Even if two participants have the same classification results, and therefore, the same $F_2$ score, the final score will also take into account a component that will penalize the mistiming of correct assignments, so that two correctly predicted nodes can contribute differently to the final metric based on the distance between the predicted node and the actual node.