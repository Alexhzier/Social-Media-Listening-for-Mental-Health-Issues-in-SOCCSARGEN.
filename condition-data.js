/**
 * Condition data for Soccsksargen (Region XII) map.
 * Edit the numbers below to change the amount shown per region per condition.
 *
 * Regions (in order):
 *   ph-sc     = South Cotabato
 *   ph-sg     = Sarangani
 *   ph-6990   = General Santos City
 */

window.conditionData = {
    'depression': [
        ['ph-sc', 12],   // South Cotabato
        ['ph-sg', 13],   // Sarangani
        ['ph-6990', 14]  // General Santos City
    ],
    'stress': [
        ['ph-sc', 14], ['ph-sg', 11], ['ph-6990', 16]
    ],
    'anxiety': [
        ['ph-sc', 13], ['ph-sg', 9], ['ph-6990', 14]
    ],
    'suicidal-behavior': [
        ['ph-sc', 3], ['ph-sg', 2], ['ph-6990', 4]
    ],
    'addiction': [
        ['ph-sc', 7], ['ph-sg', 5], ['ph-6990', 9]
    ]
};
