"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface Document {
  id: number;
  filename: string;
  upload_date: string;
  status: string;
  total_pages: number;
  text_chunks: number;
  images: number;
  tables: number;
}

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Poll for status updates if any documents are still processing
  useEffect(() => {
    const hasProcessingDocs = documents.some(
      doc => doc.status === 'pending' || doc.status === 'processing'
    );

    if (!hasProcessingDocs || loading) {
      return;
    }

    // Poll every 3 seconds
    const pollInterval = setInterval(() => {
      fetchDocuments();
    }, 3000);

    return () => clearInterval(pollInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents, loading]);

  const fetchDocuments = async () => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.warn("API request timed out after 5 seconds");
      controller.abort();
    }, 5000); // 5 second timeout
    
    try {
      setError(null);
      console.log("Fetching documents from http://localhost:8000/api/documents");
      
      const response = await fetch("http://localhost:8000/api/documents", {
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      console.log("Response received:", response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Documents loaded:", data.documents?.length || 0, "documents");
      setDocuments(data.documents || []);
    } catch (error: any) {
      clearTimeout(timeoutId);
      console.error("Error fetching documents:", error);
      console.error("Error details:", {
        name: error.name,
        message: error.message,
        stack: error.stack
      });
      
      let errorMessage = "Failed to load documents.";
      if (error.name === 'AbortError') {
        errorMessage = "Request timed out after 5 seconds. The backend may be slow or not responding. Check: 1) Is http://localhost:8000 running? 2) Check browser console (F12) for network errors.";
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setError(errorMessage);
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (id: number) => {
    if (!confirm("Are you sure you want to delete this document?")) return;
    
    try {
      await fetch(`http://localhost:8000/api/documents/${id}`, {
        method: "DELETE",
      });
      fetchDocuments();
    } catch (error) {
      console.error("Error deleting document:", error);
    }
  };

  return (
    <div className="px-4 sm:px-0">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">My Documents</h1>
        <div className="flex gap-3">
          <button
            onClick={() => {
              setLoading(true);
              fetchDocuments();
            }}
            disabled={loading}
            className="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
          <Link
            href="/upload"
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            Upload New Document
          </Link>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm font-medium text-red-800">Error: {error}</p>
          <button
            onClick={fetchDocuments}
            className="mt-2 px-4 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      )}

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
          <p className="mt-2 text-gray-600">Loading documents...</p>
          <p className="mt-1 text-xs text-gray-400">If this takes too long, check the browser console (F12) for errors</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg shadow">
          <p className="text-gray-500">No documents uploaded yet.</p>
          <Link
            href="/upload"
            className="mt-4 inline-block text-blue-600 hover:text-blue-700"
          >
            Upload your first document →
          </Link>
        </div>
      ) : (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {documents.map((doc) => (
              <li key={doc.id}>
                <div className="px-4 py-4 flex items-center sm:px-6 hover:bg-gray-50">
                  <div className="min-w-0 flex-1 sm:flex sm:items-center sm:justify-between">
                    <div className="truncate">
                      <div className="flex text-sm">
                        <p className="font-medium text-blue-600 truncate">
                          {doc.filename}
                        </p>
                        <p className="ml-2 flex-shrink-0 font-normal text-gray-500">
                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            doc.status === 'completed' ? 'bg-green-100 text-green-800' :
                            doc.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                            doc.status === 'error' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {doc.status}
                          </span>
                        </p>
                      </div>
                      <div className="mt-2 flex">
                        <div className="flex items-center text-sm text-gray-500">
                          <p>
                            {doc.total_pages} pages • {doc.text_chunks} chunks • {doc.images} images • {doc.tables} tables
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="ml-5 flex-shrink-0 flex space-x-2">
                    <Link
                      href={`/documents/${doc.id}`}
                      className="text-blue-600 hover:text-blue-900"
                    >
                      View
                    </Link>
                    <Link
                      href={`/chat?document=${doc.id}`}
                      className="text-green-600 hover:text-green-900"
                    >
                      Chat
                    </Link>
                    <button
                      onClick={() => deleteDocument(doc.id)}
                      className="text-red-600 hover:text-red-900"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
