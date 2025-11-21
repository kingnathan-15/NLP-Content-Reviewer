// Fetch sentiment data and create pie chart
fetch('/api/sentiments')
    .then(response => response.json())
    .then(evaluationData => {
        // Update stat cards
        document.getElementById('totalReviews').textContent =
            evaluationData.positive + evaluationData.neutral + evaluationData.negative;

        document.getElementById('positiveReviews').textContent = evaluationData.positive;
        document.getElementById('neutralReviews').textContent = evaluationData.neutral;
        document.getElementById('negativeReviews').textContent = evaluationData.negative;

        // Create the pie chart
        const ctx = document.getElementById('evaluationChart').getContext('2d');
        const evaluationChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    label: 'Course Evaluations',
                    data: [
                        evaluationData.positive,
                        evaluationData.neutral,
                        evaluationData.negative
                    ],
                    backgroundColor: [
                        'rgba(0, 255, 127, 0.8)',
                        'rgba(0, 191, 255, 0.8)',
                        'rgba(255, 99, 132, 0.8)'
                    ],
                    borderColor: [
                        'rgba(0, 255, 127, 1)',
                        'rgba(0, 191, 255, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 2,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 14,
                                family: "'Roboto', 'Segoe UI', sans-serif"
                            },
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(20, 30, 48, 0.95)',
                        titleColor: '#00bfff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(0, 191, 255, 0.5)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true
                }
            }
        });
    })
    .catch(error => console.error("Error fetching sentiment data:", error));

// Fetch topic distribution data and create bar chart
fetch('/api/topics')
    .then(response => response.json())
    .then(topicData => {
        const ctx = document.getElementById('topicsChart').getContext('2d');
        const topicsChart = new Chart(ctx, {
            type: 'bar',
            label: topicData.labels,
            data: {
                labels: topicData.labels,
                datasets: [{
                    label: 'Number of Reviews',
                    data: topicData.counts,
                    backgroundColor: [
                        'rgba(0, 255, 127, 0.8)',
                        'rgba(0, 191, 255, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(138, 43, 226, 0.8)'
                    ],
                    borderColor: [
                        'rgba(0, 255, 127, 1)',
                        'rgba(0, 191, 255, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(138, 43, 226, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                indexAxis: 'y', // Horizontal bar chart
                scales: {
                    x: {
                        beginAtZero: true,
                        ticks: {
                            color: '#ffffff',
                            stepSize: 1
                        },
                        grid: {
                            color: 'rgba(0, 191, 255, 0.1)'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#ffffff',
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    },
                    title: {
                        display: true,
                        text: 'Topic Distribution of Course Feedback',
                        color: '#ffffff',
                        font: {
                            size: 18,
                            family: "'Roboto','Segoe UI',sans-serif",
                            weight: '600'
                        },
                        padding: {
                            top: 5,
                            bottom: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(20, 30, 48, 0.95)',
                        titleColor: '#00bfff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(0, 191, 255, 0.5)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return `Reviews: ${context.parsed.x}`;
                            }
                        }
                    }
                }
            }
        });
    })
    .catch(error => console.error("Error fetching topic data:", error));