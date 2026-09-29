const DB_NAME = "mirai-anime-studio";
const STORE = "generations";
let connection;

function openDatabase() {
  if (!("indexedDB" in window))
    return Promise.reject(new Error("Trình duyệt không hỗ trợ IndexedDB."));
  if (!connection) {
    connection = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE))
          db.createObjectStore(STORE, { keyPath: "id" });
      };
      request.onsuccess = () => {
        const db = request.result;
        db.onversionchange = () => {
          db.close();
          connection = null;
        };
        resolve(db);
      };
      request.onerror = () => reject(request.error);
    }).catch((error) => {
      connection = null;
      throw error;
    });
  }
  return connection;
}

async function transaction(mode, callback) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const store = tx.objectStore(STORE);
    let value;
    try {
      value = callback(store);
    } catch (error) {
      tx.abort();
      reject(error);
      return;
    }
    tx.oncomplete = () => resolve(value?.result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () =>
      reject(tx.error || new Error("Không lưu được lịch sử ảnh."));
  });
}

export async function listGenerations() {
  const records = await transaction("readonly", (store) => store.getAll());
  return records.sort((a, b) => b.createdAt - a.createdAt);
}

export function saveGeneration(record) {
  return transaction("readwrite", (store) => {
    store.put(record);
    const request = store.getAll();
    // Prune inside the same transaction. Never lose track of images because a
    // UI page truncated the history, and never evict anything marked favorite.
    request.onsuccess = () => {
      let nonFavorites = 0;
      for (const item of request.result.sort(
        (a, b) => b.createdAt - a.createdAt,
      )) {
        if (item.favorite) continue;
        if (++nonFavorites > 20) store.delete(item.id);
      }
    };
    return request;
  });
}

export function deleteGeneration(id) {
  return transaction("readwrite", (store) => store.delete(id));
}

export function setFavorite(record, favorite) {
  const { url, ...stored } = record; // Object URLs are only valid for this browser session.
  return transaction("readwrite", (store) =>
    store.put({ ...stored, favorite }),
  );
}
