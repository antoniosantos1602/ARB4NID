async function deleteResult(attackId, btnElement) {
    if (!confirm("Are you sure you want to delete this result?")) return;

    try {
        const resp = await fetch(`/api/results/${attackId}`, { method: "DELETE" });
        const data = await resp.json();

        if (data.status === "ok") {
            const li = btnElement.closest("li");
            if (li) li.remove();

            const ul = document.getElementById("resultsList");
            if (ul && ul.children.length === 0) {
                ul.remove();
                const header = document.getElementById("resultsHeader");
                if (header) header.remove();
            }
        }
    } catch (err) {
        alert("Request failed: " + err);
    }
}
