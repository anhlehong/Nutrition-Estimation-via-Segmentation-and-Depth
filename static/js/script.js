const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        resultDiv.innerHTML = `
            <div class="alert alert-success">
                <h3>Results</h3>
                <p><strong>Food Type Match:</strong> ${data.food_type_match}</p>
                <p><strong>Weight:</strong> ${data.weight.toFixed(2)} grams</p>
            </div>
            <pre class="text-start bg-light p-3 rounded"><strong>Logs:</strong> ${JSON.stringify(data.logs, null, 2)}</pre>
        `;
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
});
