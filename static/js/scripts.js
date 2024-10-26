function clearTextarea() {
    document.getElementById('input-text').value = '';
}

function predictSentiment() {
    var textInput = document.getElementById('input-text').value;
    var fileInput = document.getElementById('file-input').files[0];

    if (!textInput && !fileInput) {
        alert("Please enter text or upload a file first.");
        return;
    }

    if (textInput && fileInput) {
        alert("Please enter either text or upload a file, not both.");
        return;
    }

    var data = fileInput ? { file_content: fileInput } : { text: textInput };

    if (fileInput) {
        var reader = new FileReader();
        reader.onload = function(event) {
            var fileContent = event.target.result.split('\n');
            sendRequest({ text: fileContent.join(' ') });
        };
        reader.readAsText(fileInput);
    } else {
        sendRequest(data);
    }
}

function sendRequest(data) {
    $('#loading-spinner').removeClass('d-none');
    $.ajax({
        url: 'http://127.0.0.1:5000/predict',
        type: 'POST',
        data: JSON.stringify(data),
        contentType: 'application/json',
        success: function(data) {
            $('#loading-spinner').addClass('d-none');
            if (data.error) {
                $('#error-message').text(data.error).removeClass('d-none');
                return;
            }

            $('#error-message').addClass('d-none');
            var pos = data.pos_prob || 0;
            var neg = data.neg_prob || 0;

            document.getElementById('positive-sentiment').innerText = (pos * 100).toFixed(2);
            document.getElementById('negative-sentiment').innerText = (neg * 100).toFixed(2);

            var ctx = document.getElementById('myChart').getContext('2d');
            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ["Positive", "Negative"],
                    datasets: [{
                        backgroundColor: ["green", "red"],
                        data: [pos * 100, neg * 100]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    title: {
                        display: true,
                        text: "Sentiment Analysis Results"
                    }
                }
            });

            fetchHistory();
        },
        error: function(xhr, status, error) {
            $('#loading-spinner').addClass('d-none');
            console.error("Error: " + error);
            $('#error-message').text("An error occurred while processing the text.").removeClass('d-none');
        }
    });
}

function fetchHistory() {
    $.ajax({
        url: 'http://127.0.0.1:5000/history',
        type: 'GET',
        success: function(data) {
            var historyTable = document.getElementById('history-table');
            historyTable.innerHTML = '';
            data.forEach(function(entry) {
                var row = `<tr>
                    <td>${new Date(entry.timestamp).toLocaleString()}</td>
                    <td>${entry.text}</td>
                    <td>${(entry.pos_prob * 100).toFixed(2)}%</td>
                    <td>${(entry.neg_prob * 100).toFixed(2)}%</td>
                    <td><button class="btn btn-danger btn-sm" onclick="deleteHistory(${entry.id})">Delete</button></td>
                </tr>`;
                historyTable.innerHTML += row;
            });
        },
        error: function(xhr, status, error) {
            console.error("Error: " + error);
            alert("An error occurred while fetching history.");
        }
    });
}

function deleteHistory(entryId) {
    $.ajax({
        url: `http://127.0.0.1:5000/delete_history/${entryId}`,
        type: 'DELETE',
        success: function(response) {
            if (response.error) {
                alert(response.error);
                return;
            }
            fetchHistory();
        },
        error: function(xhr, status, error) {
            console.error("Error: " + error);
            alert("An error occurred while deleting the entry.");
        }
    });
}

// Call fetchHistory when the page loads
document.addEventListener('DOMContentLoaded', function() {
    fetchHistory();
});

document.querySelectorAll('.animate-btn').forEach(button => {
    button.addEventListener('click', function (e) {
        let rect = this.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        let ripple = document.createElement('span');
        ripple.className = 'ripple';
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;
        this.appendChild(ripple);
        setTimeout(() => {
            ripple.remove();
        }, 1000);
    });
});


