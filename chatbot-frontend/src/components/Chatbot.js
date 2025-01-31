import { useState } from "react";

export default function Chatbot() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      setResponse("Erreur lors de la communication avec le serveur.");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-4">Chatbot Historique</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-md bg-white p-6 rounded-lg shadow-md">
        <input
          type="text"
          className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Posez votre question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          required
        />
        <button
          type="submit"
          className="w-full bg-blue-600 text-white p-2 mt-4 rounded-lg hover:bg-blue-700"
          disabled={loading}
        >
          {loading ? "Recherche..." : "Envoyer"}
        </button>
      </form>
      {response && (
        <div className="mt-6 bg-white p-4 rounded-lg shadow-md w-full max-w-md">
          <h2 className="text-lg font-semibold">Réponse :</h2>
          <p className="mt-2 text-gray-700">{response}</p>
        </div>
      )}
    </div>
  );
}
